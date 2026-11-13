#!/usr/bin/env bash
# Runs ON the Hyperstack VM (Ubuntu + NVIDIA driver image). Creates ~/sentinel/.venv
# and installs CUDA PyTorch + project requirements. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."

TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"

nvidia-smi || { echo "ERROR: no NVIDIA driver on this VM - pick a GPU flavor with a CUDA image"; exit 1; }

need=(); for p in python3-venv python3-pip tmux; do dpkg -s "$p" >/dev/null 2>&1 || need+=("$p"); done
if [ ${#need[@]} -gt 0 ]; then
  APT="sudo DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600"
  $APT update -y && $APT install -y "${need[@]}"
fi

[ -x .venv/bin/pip ] || { rm -rf .venv; python3 -m venv .venv; }
. .venv/bin/activate
pip install --upgrade pip -q
pip install -q --timeout 300 torch==2.8.0 --index-url "$TORCH_INDEX"
pip install -q --timeout 300 -r requirements.txt
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m sentinel.data   # pre-download SFT splits from Supabase
echo "VM ready."
