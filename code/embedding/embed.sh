#!/bin/bash
#SBATCH --job-name=crysjepa-embed
#SBATCH --partition=gpu        # GPU nodes (xgp*); 'normal' is CPU-only. Use gpu-long if >3h.
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00        # gpu partition max is 3:00:00
#SBATCH --output=embed_%j.log
#
# Submit with:  sbatch embed.sh      (check logs: embed_<jobid>.log)
#
# One-time, before first run, create the env (SoC has no conda/modules; use venv):
#   /usr/bin/python3.12 -m venv ~/envs/crysjepa
#   source ~/envs/crysjepa/bin/activate
#   export TMPDIR=/tmp/$USER && mkdir -p "$TMPDIR"
#   pip install --no-cache-dir torch torch_geometric pymatgen huggingface_hub pandas pyyaml tqdm

source ~/envs/crysjepa/bin/activate
export TMPDIR=/tmp/$USER && mkdir -p "$TMPDIR"

# get the code + data once (skipped if already present)
[ -d Crys_JEPA ] || git clone https://github.com/liun-online/Crys_JEPA.git
if [ ! -d cj/exp_logs ]; then
  python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("liun-online/Crys_JEPA", repo_type="dataset",
                  local_dir="cj", allow_patterns=["data/*", "exp_logs.zip"])
PY
  (cd cj && unzip -q -o exp_logs.zip)
fi

python embed.py
