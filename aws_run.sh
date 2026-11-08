#!/usr/bin/env bash
# =============================================================================
# AWS FINETUNE RUN SCRIPT
# Run the SENTINEL fraud-detection pipeline on an AWS EC2 (g4dn.xlarge / p3.2xlarge)
#
# Usage:
#   bash aws_run.sh                          # LoRA mode (default)
#   bash aws_run.sh --mode qlora             # Q-LoRA mode (4-bit quantization)
#   bash aws_run.sh --mode dry-run           # Validate data pipeline only
#   bash aws_run.sh --mode eval              # Skip training, evaluate only
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration - override these with environment variables or edit directly
# ---------------------------------------------------------------------------
SUPABASE_URL="${SUPABASE_URL:-}"
SUPABASE_KEY="${SUPABASE_KEY:-}"          # MUST be set before running
MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/sentinel-lora}"
MAX_RECORDS="${MAX_RECORDS:-50000}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
LR="${LR:-2e-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"

# Load .env if available
if [ -f .env ]; then
  set -a
  source .env
  set +a
  echo ">> Loaded .env"
fi

# Mode selection (first arg or default to "full")
MODE="${1:-full}"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
if [ -z "$SUPABASE_KEY" ]; then
  echo "ERROR: SUPABASE_KEY is not set."
  echo "Set it via environment: export SUPABASE_KEY=your_supabase_service_key"
  echo "Or edit this script."
  exit 1
fi

# Detect GPU
if command -v nvidia-smi &> /dev/null; then
  GPU_COUNT=$(nvidia-smi --query-gpu=name,count --format=csv,noheader | cut -d',' -f2)
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d '[:space:]')
  echo "GPU: $GPU_NAME (x${GPU_COUNT})"
else
  echo "WARNING: nvidia-smi not found. Ensure you are on a GPU instance (g4dn.*, p3.*)."
  echo "Falling back to CPU (will be very slow for LLM fine-tuning)."
fi

# ---------------------------------------------------------------------------
# Install dependencies
# ---------------------------------------------------------------------------
echo ">> Installing Python dependencies ..."
pip install --upgrade pip
pip install -r requirements.txt
pip install --quiet --upgrade transformers peft accelerate bitsandbytes

# ---------------------------------------------------------------------------
# Create output directories
# ---------------------------------------------------------------------------
mkdir -p "$OUTPUT_DIR" "./data/cache" "./results"

# ---------------------------------------------------------------------------
# Build the main.py arguments
# ---------------------------------------------------------------------------
PY_ARGS=()

case "$MODE" in
  dry-run)
    PY_ARGS+=(--dry-run)
    ;;
  eval)
    PY_ARGS+=(--mode eval)
    ;;
  train)
    PY_ARGS+=(--mode train)
    ;;
  full)
    PY_ARGS+=(--mode full)
    ;;
  qlora)
    export LORA_MODE="qlora"
    PY_ARGS+=(--mode full)
    ;;
  lora)
    export LORA_MODE="lora"
    PY_ARGS+=(--mode full)
    ;;
  *)
    echo "ERROR: Unknown mode '$MODE'. Use: full, train, eval, dry-run, lora, qlora"
    exit 1
    ;;
esac

# ---------------------------------------------------------------------------
# Export environment variables that config.py reads
# ---------------------------------------------------------------------------
export SUPABASE_URL
export SUPABASE_KEY
export MODEL_NAME
export MAX_RECORDS
export NUM_EPOCHS
export LR
export BATCH_SIZE
export OUTPUT_DIR

echo ""
echo "============================================"
echo "  SENTINEL - AWS FINETUNE"
echo "============================================"
echo "  Mode       : $MODE"
echo "  Model      : $MODEL_NAME"
echo "  Records    : $MAX_RECORDS"
echo "  Epochs     : $NUM_EPOCHS"
echo "  LR         : $LR"
echo "  Batch size : $BATCH_SIZE"
echo "  Output     : $OUTPUT_DIR"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# Run the pipeline
# ---------------------------------------------------------------------------
python main.py "${PY_ARGS[@]}"

# ---------------------------------------------------------------------------
# Post-run: summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
echo "  FINETUNE COMPLETE"
echo "============================================"

if [ -f "$OUTPUT_DIR/adapter/config.json" ]; then
  echo "  Adapter saved: $OUTPUT_DIR/adapter/"
  du -sh "$OUTPUT_DIR/adapter/"
fi

if [ -f "$OUTPUT_DIR/evaluation_report.json" ]; then
  echo "  Evaluation report: $OUTPUT_DIR/evaluation_report.json"
  echo ""
  echo "  --- Metrics ---"
  cat "$OUTPUT_DIR/evaluation_report.json" | python -m json.tool 2>/dev/null || cat "$OUTPUT_DIR/evaluation_report.json"
fi

if [ -f "$OUTPUT_DIR/metrics.json" ]; then
  echo ""
  echo "  --- Training metrics ---"
  cat "$OUTPUT_DIR/metrics.json" | python -m json.tool 2>/dev/null || cat "$OUTPUT_DIR/metrics.json"
fi

echo ""
echo "============================================"
echo "  DONE - Files are in $OUTPUT_DIR"
echo "============================================"
