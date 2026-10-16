"""
Crys-JEPA embeddings for MP-20, to run on the SoC GPU cluster.

Produces out/embeddings.npy (N x 512) + out/metadata.csv (row i <-> row i).
Assumes the repo + HF data are already present (embed.sh handles that).
Run it on a GPU node (see embed.sh).
"""
import os, sys, glob
import numpy as np, pandas as pd, torch, torch.nn.functional as F, yaml
from types import SimpleNamespace
from pymatgen.core.structure import Structure
from tqdm import tqdm

REPO = "Crys_JEPA"      
DATA = "cj"             # the downloaded + unzipped HF data
OUT  = "out"; os.makedirs(OUT, exist_ok=True)

N_LIMIT    = None      
BATCH      = 256
SCALER_CAP = 20000      # crystals per file used to fit the input scaler

sys.path.insert(0, REPO)
from components.jepa.frame.jepa import JEPA

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)


# --- preprocessing: match the model's training pipeline exactly ---
def type_sort(frac, z):
    key = 1000 * z + 100 * frac[:, 0] + 10 * frac[:, 1] + frac[:, 2]
    return torch.argsort(key)

def polar6(cell):                         # 3x3 lattice -> 6-number descriptor
    W, S, Vt = torch.linalg.svd(cell)
    U = W @ Vt
    sym = U @ (Vt.T @ torch.diag(S) @ Vt) @ U.T
    sym[sym.abs() < 1e-5] = 0
    i, j = torch.triu_indices(3, 3)
    return sym[i, j]

def prep(structure):                      # primitive + reduced + atom-sorted
    s = structure.get_primitive_structure().get_reduced_structure()
    frac = torch.tensor(s.frac_coords, dtype=torch.float32)
    z = torch.tensor(s.atomic_numbers, dtype=torch.float32)
    order = type_sort(frac, z)
    cell = torch.tensor(s.lattice.matrix, dtype=torch.float32)
    return frac[order], z[order].long(), polar6(cell), len(z)

@torch.no_grad()
def embed(batch, model, means, stds):     # batch = list of prep() tuples
    frac = torch.cat([b[0] for b in batch]).to(device)
    z    = torch.cat([b[1] for b in batch]).to(device)
    m6   = torch.stack([b[2] for b in batch])
    nat  = torch.tensor([b[3] for b in batch])
    scaled = ((m6 / nat[:, None] ** (1/3) - means) / stds).to(device)
    idx = torch.repeat_interleave(torch.arange(len(batch), device=device), nat.to(device))
    x = torch.cat([frac, F.one_hot(z, 100).float(), scaled[idx]], -1)   # 3+100+6 = 109
    return model.encode(x, idx).cpu().numpy()


# --- find the files ---
def find(pattern):
    return sorted(glob.glob(f"{DATA}/**/{pattern}", recursive=True))[-1]

train = find("base/mp_20/train.csv.gz")
mp, mptrj = find("jepa/mp.csv.gz"), find("jepa/mptrj.csv.gz")
ckpt = sorted(glob.glob(f"{DATA}/**/jepa/saved_model/model_*.pt", recursive=True),
              key=lambda p: int(p.split("_")[-1].split(".")[0]))[-1]
print("ckpt:", ckpt)


# --- fit the input scaler (not shipped; recompute from the training crystals) ---
scaler_file = f"{OUT}/scaler.pt"
if os.path.exists(scaler_file):
    means, stds = torch.load(scaler_file)
else:
    mats, nats = [], []
    for f in [mp, mptrj]:
        df = pd.read_csv(f, compression="gzip")
        if SCALER_CAP:
            df = df.sample(min(SCALER_CAP, len(df)), random_state=0)
        for cif in tqdm(df["cif"], desc=f"scaler {os.path.basename(f)}"):
            try:
                _, _, m6, nat = prep(Structure.from_str(cif, fmt="cif"))
            except Exception:
                continue
            mats.append(m6); nats.append(nat)
    scaled = torch.stack(mats) / torch.tensor(nats)[:, None] ** (1/3)
    means, stds = scaled.mean(0), scaled.std(0)
    torch.save((means, stds), scaler_file)
print("scaler ready")


# --- load the model ---
cfg = yaml.safe_load(open(f"{REPO}/configs/jepa/mp.yml"))
config = SimpleNamespace(model=SimpleNamespace(**cfg["model"]), device=device)
model = JEPA(config)
state = torch.load(ckpt, map_location=device, weights_only=False)["model_state_dict"]
model.load_state_dict({k.replace("module.", "", 1): v for k, v in state.items()}, strict=False)
model.eval().to(device)
print("model ready")


# --- embed the MP-20 reference crystals ---
df = pd.read_csv(train, compression="gzip")
if N_LIMIT:
    df = df.iloc[:N_LIMIT]

embs, kept, buf = [], [], []
for i, cif in tqdm(list(zip(df.index, df["cif"])), desc="embed"):
    try:
        buf.append((i, prep(Structure.from_str(cif, fmt="cif"))))
    except Exception:
        continue
    if len(buf) == BATCH:
        embs.append(embed([p for _, p in buf], model, means, stds))
        kept += [i for i, _ in buf]; buf = []
if buf:
    embs.append(embed([p for _, p in buf], model, means, stds))
    kept += [i for i, _ in buf]

emb = np.vstack(embs).astype(np.float32)
meta = df.loc[kept].drop(columns=[c for c in ["cif", "Unnamed: 0"] if c in df], errors="ignore")
np.save(f"{OUT}/embeddings.npy", emb)
meta.to_csv(f"{OUT}/metadata.csv", index=False)
print("done:", emb.shape, "saved to", OUT)
