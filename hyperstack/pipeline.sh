#!/usr/bin/env bash
# Run ON the Hyperstack VM. The whole model side in ONE command, unattended (runs in tmux, survives SSH drops):
#   setup -> health check -> train LoRA -> train QLoRA -> train XGBoost -> serve all three -> print the AWS settings
#
#   bash hyperstack/pipeline.sh               # full training (~20-30 min on one GPU)
#   QUICK=1 bash hyperstack/pipeline.sh       # short demo training (~5 min)
#   SKIP_TRAIN=1 bash hyperstack/pipeline.sh  # VM restarted: only (re)serve the models that are already trained
#
# Follow:  tail -f output/pipeline.log        Result: output/aws.env (2 lines to paste into .env on your PC)
# Firewall (Hyperstack console): inbound TCP 22, 8000, 8001, 8002.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p output
LOG=output/pipeline.log

if [ "${SENTINEL_PIPELINE_INSIDE:-0}" != "1" ]; then
  command -v tmux >/dev/null || { sudo apt-get update -y && sudo apt-get install -y tmux; }
  if tmux has-session -t pipeline 2>/dev/null; then
    echo "The pipeline is already running.  Follow: tail -f $LOG   (stop: tmux kill-session -t pipeline)"; exit 1
  fi
  : > "$LOG"
  tmux new-session -d -s pipeline \
    "SENTINEL_PIPELINE_INSIDE=1 QUICK=${QUICK:-0} SKIP_TRAIN=${SKIP_TRAIN:-0} bash hyperstack/pipeline.sh"
  echo "Pipeline started in tmux session 'pipeline' (safe to disconnect)."
  echo "  follow : tail -f $LOG"
  echo "  done   : the log ends with 'PIPELINE DONE' and output/aws.env holds the settings for your PC"
  exit 0
fi

exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "=================== [$(date '+%T')] $*"; }
die()  { echo "PIPELINE FAILED at: $*"; exit 1; }

EXTRA=""; [ "${QUICK:-0}" = "1" ] && EXTRA="--max-steps 30 --max-eval-rows 2000"

step "1/6 setup (venv, CUDA torch, deps, data, Qwen weights)"
bash hyperstack/setup_vm.sh || die "setup_vm.sh (is .env filled with SUPABASE_URL / SUPABASE_KEY?)"

step "2/6 health check"
bash hyperstack/check.sh || echo "(check reported problems; continuing - services not running yet is expected)"

if [ "${SKIP_TRAIN:-0}" != "1" ]; then
  step "3/6 train LoRA on the GPU"
  .venv/bin/python -m sentinel.train --tag hyperstack --method lora $EXTRA || die "LoRA training"
  step "4/6 train QLoRA on the GPU"
  .venv/bin/python -m sentinel.train --tag hyperstack --method qlora $EXTRA || die "QLoRA training"
  step "5/6 train XGBoost on the GPU"
  .venv/bin/python xgboost_benchmark/train.py --device cuda --out output/xgb || die "XGBoost training"
else
  step "3-5/6 training skipped (SKIP_TRAIN=1)"
fi

step "6/6 serve LoRA :8000, XGBoost :8001, QLoRA :8002"
bash hyperstack/serve_all.sh || die "serve_all.sh"
bash hyperstack/check.sh || true

IP="$(curl -fs --max-time 3 https://api.ipify.org || echo '<VM_PUBLIC_IP>')"
KEY="$(grep '^SENTINEL_API_KEY=' .env | tail -1 | cut -d= -f2- | tr -d '\r')"
printf 'TF_VAR_hyperstack_host=%s\nTF_VAR_model_api_key=%s\n' "$IP" "$KEY" > output/aws.env
chmod 600 output/aws.env
echo
echo "PIPELINE DONE. Models are served and auto-restart if they crash."
echo "Copy these 2 lines into .env on your PC (repo root), then run:  python scripts/ops.py up"
echo "  (they are also in output/aws.env on this VM)"
echo "    TF_VAR_hyperstack_host=$IP"
echo "    TF_VAR_model_api_key=<the value in output/aws.env>"
