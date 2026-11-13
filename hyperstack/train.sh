#!/usr/bin/env bash
# Run ON the VM. Starts training in a tmux session named "train" (survives SSH disconnects).
#   bash hyperstack/train.sh                    # LoRA defaults
#   bash hyperstack/train.sh --method qlora --epochs 3
# Watch:  tail -f output/last_train.log      or  tmux attach -t train  (detach: Ctrl-B then D)
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x .venv/bin/python ] || { echo "Run: bash hyperstack/setup_vm.sh first"; exit 1; }
mkdir -p output
tmux kill-session -t train 2>/dev/null || true
args=$(printf '%q ' "$@")
tmux new-session -d -s train \
  "source .venv/bin/activate && python -m sentinel.train --tag hyperstack $args 2>&1 | tee output/last_train.log"
echo "Training started (tmux session 'train'). Follow with: tail -f output/last_train.log"
