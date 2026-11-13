# SENTINEL — fraud-scoring LLM, XGBoost benchmark, and dashboard

```
 Hyperstack GPU VM                 Your machine                         Anyone with the links
 ┌───────────────────────┐        ┌────────────────────────────┐       ┌──────────────────────┐
 │ LLM API  :8000        │        │ XGBoost API :8001 ──ngrok──┼──────▶│ Streamlit dashboard  │
 │ Qwen2.5-0.5B + LoRA   │◀───────┼────────────────────────────┼───────│ :8501 (local, Docker │
 │ sentinel/ + hyperstack│  HTTP  │ xgboost_benchmark/         │       │  or any host)  ui/   │
 └───────────────────────┘        └────────────────────────────┘       └──────────────────────┘
        Dockerfile.llm                xgboost_benchmark/Dockerfile          ui/Dockerfile
```

All three services share one contract: `POST /score` takes `{"prompt"}` or structured fields or
`{"transactions":[...]}` and returns
`{transaction_id, fraud_probability, decision: ALLOW|REVIEW|BLOCK, confidence, reason_codes, explanation}`.
`GET /stats` returns metrics (incl. PR/ROC curves), calibration and model info. Auth uses the `X-API-Key` header.

| Folder | What | Runs on |
|---|---|---|
| `sentinel/`, `hyperstack/`, `tests/`, `Dockerfile.llm` | LLM training (LoRA/QLoRA), eval, GPU inference API | Hyperstack GPU |
| `xgboost_benchmark/` | XGBoost baseline: same splits, prompts, and protocol; API with samples and scenarios | Your machine (via ngrok) |
| `ui/` | iOS-style Streamlit dashboard: stats, benchmark, live and batch tests | Anywhere |

Both models use the same protocol. They train on `train` (Jan–Oct 2025), then early-stop, calibrate, and pick
thresholds on `validation` (REVIEW: recall ≥ 0.85, BLOCK: precision ≥ 0.70). Holdout metrics (Jan–Feb 2026)
use those frozen thresholds. Data comes from the Supabase bucket `sentinel-data/training/sft/*`.

---

## A. LLM on the Hyperstack GPU

**Console:** single-GPU NVIDIA VM, an Ubuntu image with NVIDIA drivers/CUDA, your SSH key, public IP.
Open inbound TCP **22** and **8000**.

```bash
ssh ubuntu@<VM_IP>
git clone https://github.com/euron-tech/Project-sentinal-gpu-training-of-model.git sentinel && cd sentinel
cp .env.example .env && nano .env                # SUPABASE_URL, SUPABASE_KEY
bash hyperstack/setup_vm.sh                      # venv, CUDA torch, deps, data -> "VM ready."
source .venv/bin/activate
python -m pytest -q tests                        # 15 passed (LoRA + QLoRA + HTTP smoke on the GPU)

bash hyperstack/train.sh --method lora           # or --method qlora; runs in tmux
tail -f output/last_train.log                    # done at "Artifacts in output/<run>"

RUN=$(ls -1dt output/*/ | head -1)
python -m sentinel.evaluate --run $RUN --split holdout
python -m sentinel.infer --run $RUN sample --split holdout --n 5 --fraud-only

PUBLIC_IP=<VM_IP> bash hyperstack/serve.sh       # GPU API on :8000 in tmux; prints URL + API key
```
Check from your PC: `curl http://<VM_IP>:8000/health` (should show `"device": "cuda:0"`).
To update: `git pull && bash hyperstack/setup_vm.sh`, retrain or re-serve. **Hibernate the VM when done.**

## B. XGBoost benchmark on your machine (Windows/Linux)

The model is already trained and committed (`xgboost_benchmark/artifacts/`, holdout PR-AUC 0.215, ROC-AUC 0.973).
To keep it running all the time (Docker, auto-restart; prints the URL and API key):
```powershell
powershell -ExecutionPolicy Bypass -File xgboost_benchmark\run_service.ps1
ngrok http 8001                                  # share the https URL + key with the dashboard
```
Or retrain or run it without Docker:
```bash
# from the repo root, with the venv active (or: pip install -r xgboost_benchmark/requirements.txt)
cd xgboost_benchmark
python train.py                                  # ~1 min CPU; writes artifacts/ (already committed)
python -m pytest -q tests                        # 5 passed
# PowerShell:  $env:XGB_API_KEY = "<pick a long random key>"     bash:  export XGB_API_KEY=...
uvicorn server:app --host 127.0.0.1 --port 8001
```
Expose it in a second terminal:
```bash
ngrok http 8001                                  # copy the https://<id>.ngrok-free.app URL
curl https://<id>.ngrok-free.app/health
```

## C. Dashboard

```bash
pip install -r ui/requirements.txt               # already in the root .venv
# PowerShell: $env:LLM_URL="http://<VM_IP>:8000"; $env:LLM_API_KEY="<from serve.sh>"
#             $env:XGB_URL="https://<id>.ngrok-free.app"; $env:XGB_API_KEY="<your key>"
streamlit run ui/app.py                          # http://localhost:8501 (URLs/keys can also be set in the sidebar)
python -m pytest -q ui/tests                     # headless render test (set UI_TEST_* for the live variant)
```
Tabs: **Overview** (holdout KPIs and acceptance targets OBJ-01/02/03/04/10) · **LLM** (run info, loss, thresholds,
reason-code F1) · **Benchmark** (PR/ROC overlays, operating points, uplift, feature importance) · **Live Test**
(scenario presets, a custom form, or a raw prompt, scored on both models with latency) · **Batch Test** (labelled
samples from the XGBoost service or an uploaded JSONL/CSV, with sample metrics and a CSV export).

To share the dashboard with others: run `ngrok http 8501` too, or deploy the `ui/` Docker image anywhere.

## Docker (each service deploys on its own)

```bash
docker build -t sentinel-xgb xgboost_benchmark   # XGBoost API (model baked in)
docker run -d -p 8001:8001 -e XGB_API_KEY=... -v "$PWD/data/sft:/data/sft" sentinel-xgb

docker build -t sentinel-ui ui                   # dashboard
docker run -d -p 8501:8501 -e LLM_URL=http://<VM_IP>:8000 -e LLM_API_KEY=... \
           -e XGB_URL=https://<id>.ngrok-free.app -e XGB_API_KEY=... sentinel-ui

docker build -f Dockerfile.llm -t sentinel-llm . # LLM API, on a GPU host with NVIDIA Container Toolkit
docker run -d --gpus all -p 8000:8000 -e SENTINEL_API_KEY=... -v "$PWD/output:/app/output" \
           sentinel-llm --run output/<run> serve --host 0.0.0.0 --port 8000
```
Or use compose: `docker compose up -d --build xgboost ui` (your machine), then
`docker compose --profile gpu up -d --build llm` (GPU host, set `LLM_RUN=<run>` in `.env`).

## Troubleshooting
| Symptom | Fix |
|---|---|
| Dashboard shows LLM offline | VM firewall needs TCP 8000; `tmux ls` on the VM should list `serve`; check the key |
| XGBoost offline via ngrok | Is `uvicorn` still running? Is the ngrok URL current (it changes on restart)? Is `XGB_API_KEY` the same? |
| 401 | Wrong or missing API key in the sidebar |
| `setup_vm.sh`: `cuda False` | Image driver too old. Use a newer image, or `TORCH_INDEX=https://download.pytorch.org/whl/cu128` |
| CUDA OOM while training | `--batch-size 8 --grad-accum 2` or `--method qlora` |

Plain HTTP with shared keys is for testing only. Put the APIs behind HTTPS before using real data.
