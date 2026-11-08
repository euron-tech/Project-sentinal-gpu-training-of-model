# ============================================================
# SENTINEL - DOCKERFILE (LOCAL / CPU RUN)
#
# Usage:
#   # Build the image
#   docker build -t sentinel-finetune:latest .
#
#   # Dry-run (data pipeline only, no ML libraries needed at run-time)
#   docker run --rm \
#     -e SUPABASE_URL="https://mdizvmguddpnvanfbegn.supabase.co" \
#     -e SUPABASE_KEY="sb_publishable_EUMi6FM_AdDt0zp71yfsNA_ntUEZArr" \
#     sentinel-finetune:latest --dry-run
#
#   # Full pipeline (needs HF token for gated models)
#   docker run --rm \
#     -e SUPABASE_URL="https://mdizvmguddpnvanfbegn.supabase.co" \
#     -e SUPABASE_KEY="sb_publishable_EUMi6FM_AdDt0zp71yfsNA_ntUEZArr" \
#     -e HUGGING_FACE_HUB_TOKEN="hf_..." \
#     -v "${PWD}/output:/app/output" \
#     -v "${PWD}/results:/app/results" \
#     sentinel-finetune:latest --mode full
#
#   # Using docker-compose (recommended)
#     docker compose run --rm sentinel --dry-run
# ============================================================

# --- Stage 1: Build (install wheels) ---
FROM python:3.11-slim AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .

# Pin torch 2.3.1 (CPU-only, last version compatible with accelerate<0.30)
# Set a long pip timeout (large wheels on slow connections) and use PyPI as fallback index.
# Each step is a separate cache layer so progress is not lost on timeout.

# 1. Core ML framework
RUN pip config set global.timeout 600 && \
    pip install --no-cache-dir --prefix="/opt/venv" \
    torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu

# 2. Create constraint file to prevent later installs from upgrading torch or transformers
RUN echo "torch==2.3.1" > /build/torch-constraints.txt && \
    echo "transformers==4.37.0" >> /build/torch-constraints.txt && \
    echo "huggingface-hub<1.0" >> /build/torch-constraints.txt

# 3. Install ALL remaining packages in one step with strict constraints
# This prevents peft/accelerate from upgrading transformers/hf-hub during resolution
RUN pip install --no-cache-dir --prefix="/opt/venv" \
    --constraint /build/torch-constraints.txt \
    transformers==4.37.0 \
    huggingface-hub==0.20.0 \
    peft==0.6.0 \
    accelerate==0.25.0 \
    datasets==2.15.0 \
    pandas \
    numpy \
    supabase \
    requests \
    scikit-learn \
    python-dotenv \
    sentencepiece

# 4. Lock versions: re-install critical packages to ensure no dependency drift
# --no-deps avoids re-resolving dependencies that might upgrade things
RUN pip install --no-cache-dir --prefix="/opt/venv" \
    --no-deps \
    transformers==4.37.0 \
    huggingface-hub==0.20.0 \
    peft==0.6.0


# --- Stage 2: Runtime ---
FROM python:3.11-slim

# Copy the virtualenv from builder
COPY --from=builder /opt/venv /opt/venv

# Make venv the default Python (add executables + site-packages to Python path)
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/opt/venv/lib/python3.11/site-packages"
ENV PYTHONUNBUFFERED=1

# Install libmagic (optional, for file type detection)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libmagic1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy application code
COPY . .

# Create default output directories
RUN mkdir -p /app/output /app/results

# Default environment variables
ENV SUPABASE_URL="https://mdizvmguddpnvanfbegn.supabase.co"
ENV SUPABASE_BUCKET="sentinel-data"
ENV MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
ENV MAX_RECORDS="50000"
ENV NUM_EPOCHS="3"
ENV LR="2e-4"
ENV BATCH_SIZE="4"
ENV OUTPUT_DIR="./output/sentinel-lora"

# Print system info at startup
RUN echo "SENTINEL pipeline ready." && \
    python -c "import torch; print(f'PyTorch {torch.__version__} (CPU)')" || true

CMD ["python", "main.py"]
