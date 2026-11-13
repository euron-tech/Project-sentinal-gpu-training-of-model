# SENTINEL — LoRA / QLoRA fraud scoring on Qwen2.5 (Hyperstack GPU)

Fine-tunes `Qwen/Qwen2.5-0.5B-Instruct` to read a serialized transaction plus the customer's
recent history and return:

```json
{"transaction_id": "TX_…", "fraud_probability": 0.91, "decision": "BLOCK",
 "confidence": 0.91, "reason_codes": ["R01_NEW_DEVICE", "R03_ODD_HOUR"],
 "explanation": "Blocked: new device, unusual hour."}
```

The LLM gets a 9-way head: one fraud logit (Platt-calibrated on validation) plus eight reason-code
logits. LoRA goes on all linear layers. QLoRA loads the base in 4-bit NF4. Thresholds are
picked on validation (REVIEW: recall ≥ 0.85, BLOCK: precision ≥ 0.70), then frozen for holdout and serving.

Data: the prebuilt splits in the Supabase bucket `sentinel-data/training/sft/{train,validation,holdout}`
(train Jan–Oct 2025, validation Nov–Dec 2025, holdout Jan–Feb 2026, ~0.6 % fraud), downloaded to `data/sft/`.

```
sentinel/    data.py  model.py  train.py  evaluate.py  infer.py (CLI + HTTP server)
hyperstack/  setup_vm.sh  train.sh  serve.sh   (run ON the VM)
             remote.sh                         (optional: drive the VM from your PC over SSH)
tests/       unit tests + tiny-model LoRA/QLoRA train → eval → infer → HTTP smoke test
```

---

## Run everything on a Hyperstack GPU VM

### 0. Launch the VM (Hyperstack console)
- Any single-GPU NVIDIA flavor, an Ubuntu image with NVIDIA drivers/CUDA, your SSH key, public IP on.
- Firewall rules, inbound TCP: **22** (SSH) and **8000** (test API). Restrict 8000 to your testers' IPs if you can.

### 1. SSH in and clone
```bash
ssh ubuntu@<VM_IP>
nvidia-smi                                   # GPU must be listed
git clone https://github.com/euron-tech/Project-sentinal-gpu-training-of-model.git sentinel
cd sentinel
```

### 2. Create `.env` on the VM (never committed)
```bash
cp .env.example .env
nano .env        # set SUPABASE_URL and SUPABASE_KEY (SUPABASE_BUCKET=sentinel-data)
```

### 3. Install + download data (one time, ~5–10 min)
```bash
bash hyperstack/setup_vm.sh                  # ends with: torch … cuda True <GPU> / VM ready.
source .venv/bin/activate
```

### 4. Test the pipeline on the GPU (~1 min)
```bash
python -m pytest -q                          # expect: 15 passed
```

### 5. Train
```bash
bash hyperstack/train.sh --method lora       # or --method qlora; any sentinel.train flag works
tail -f output/last_train.log                # Ctrl-C stops only the tail; training keeps running
```
Training is done when the log prints `Holdout: {...}` and `Artifacts in output/<run>`.
Each run folder holds `adapter/`, `calibration.json`, `metrics_validation.json`, `metrics_holdout.json`, `train_log.json`.

### 6. Evaluate + inference on the VM GPU
```bash
RUN=$(ls -1dt output/*/ | head -1)
python -m sentinel.evaluate --run $RUN --split holdout
python -m sentinel.infer --run $RUN sample --split holdout --n 5
python -m sentinel.infer --run $RUN sample --split holdout --n 5 --fraud-only
python -m sentinel.infer --run $RUN latency --n 50
```

### 7. Make it available to testers
```bash
PUBLIC_IP=<VM_IP> bash hyperstack/serve.sh   # GPU server on 0.0.0.0:8000 in tmux, prints the API key
```
Share the URL and the API key with testers:
- **Browser:** `http://<VM_IP>:8000/`. Paste the key and press **Score**.
- **curl:**
  ```bash
  curl -s http://<VM_IP>:8000/health
  curl -s http://<VM_IP>:8000/score -H "X-API-Key: <KEY>" -H "Content-Type: application/json" \
    -d '{"transaction_id":"TX1","amount":75000,"hour":3,"new_device":true,"fails_24h":4,"account_age_days":6}'
  curl -s http://<VM_IP>:8000/score -H "X-API-Key: <KEY>" -H "Content-Type: application/json" \
    -d '{"transactions":[{"amount":250,"hour":14},{"amount":90000,"hour":2,"new_device":true}]}'
  ```
- **Python:** `requests.post(url + "/score", json=txn, headers={"X-API-Key": KEY}).json()`

Input fields: `transaction_id, amount, mcc, channel, hour, weekday, customer_age, account_age_days,
avg_90d, txns_30d, fails_24h, new_device, city, merchant_city, new_city, dist_km, amt_ratio, txns_1h,
amount_24h, device_users_7d, history[{days_ago, channel, amount, mcc, city}]`, or `{"prompt": "<fmt-v1 text>"}`.

Server control: `tmux attach -t serve` (detach Ctrl-B D) · `tail -f output/serve.log` · stop with `tmux kill-session -t serve`.
Plain HTTP with a shared key is meant for testing only. Put it behind HTTPS before using it for anything real.

### 8. Update code / stop
```bash
git pull && bash hyperstack/setup_vm.sh      # after pushing new code
```
**Hibernate or delete the VM in the console when finished.** It bills by the hour.

---

## Troubleshooting
| Symptom | Fix |
|---|---|
| `setup_vm.sh`: `cuda False` / torch import error | Driver too old for cu126 wheels. Use an image with a newer driver, or `TORCH_INDEX=https://download.pytorch.org/whl/cu128 bash hyperstack/setup_vm.sh` |
| `SUPABASE_URL and SUPABASE_KEY must be set` | Fill in `.env` in the repo folder |
| CUDA out of memory | `--batch-size 8 --grad-accum 2`, or `--method qlora` |
| apt lock during setup | Fresh VM still self-updating. Wait, then re-run |
| Testers can't reach :8000 | Add the inbound TCP 8000 firewall rule; check `tmux ls` shows `serve` |
| 401 from /score | Missing or wrong `X-API-Key`; see `grep SENTINEL_API_KEY .env` on the VM |

## Local GPU (optional)
```bash
python -m venv .venv && .venv/Scripts/activate      # Linux: source .venv/bin/activate
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt && python -m pytest -q
```
`hyperstack/remote.sh` can also push, set up, train, and pull from your PC over SSH; see the header of that file.
