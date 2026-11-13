# SENTINEL — LoRA / QLoRA fraud scoring on Qwen2.5

Fine-tunes `Qwen/Qwen2.5-0.5B-Instruct` to read a serialized transaction plus the
customer's recent history (fmt-v1 prompt) and return

```json
{"transaction_id": "TX_…", "fraud_probability": 0.91, "decision": "BLOCK",
 "confidence": 0.91, "reason_codes": ["R01_NEW_DEVICE", "R03_ODD_HOUR"],
 "explanation": "Blocked: new device, unusual hour."}
```

The LLM gets a 9-way head: one fraud logit (BCE, Platt-calibrated on validation)
and eight reason-code logits. LoRA goes on all linear layers. QLoRA loads the base in 4-bit NF4.
REVIEW and BLOCK thresholds come from validation (recall ≥ 0.85 → REVIEW,
precision ≥ 0.70 → BLOCK), then stay frozen for holdout and inference.

Data: the prebuilt splits in the Supabase bucket `sentinel-data/training/sft/{train,validation,holdout}`
(train Jan–Oct 2025, validation Nov–Dec 2025, holdout Jan–Feb 2026, ~0.6 % fraud).
They download automatically to `data/sft/` on first use.

```
sentinel/   data.py  model.py  train.py  evaluate.py  infer.py
hyperstack/ setup_vm.sh (runs on VM)   remote.sh (drives VM over SSH)
tests/      unit tests + tiny-model end-to-end smoke test
```

## 1. Setup (local GPU, Windows or Linux)

```bash
cp .env.example .env            # fill SUPABASE_URL / SUPABASE_KEY
python -m venv .venv
.venv/Scripts/activate          # Linux: source .venv/bin/activate
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -m sentinel.data         # download the SFT splits (~13 MB)
python -m pytest -q             # unit + smoke tests
```

## 2. Train locally

```bash
python -m sentinel.train                              # LoRA, 2 epochs, 10 genuine : 1 fraud
python -m sentinel.train --method qlora               # 4-bit QLoRA
python -m sentinel.train --max-steps 30 --max-eval-rows 2000   # 1–2 min sanity run
```
Useful flags: `--epochs --lr --batch-size --grad-accum --lora-r --lora-alpha --max-length
--genuine-per-fraud (0 = all rows) --pos-weight --eligible-only --base-model`.

Each run writes `output/<timestamp>-<method>-<tag>/` with `adapter/`, `calibration.json`,
`metrics_validation.json`, `metrics_holdout.json`, `train_log.json`, `train.log`.

## 3. Train on Hyperstack

1. In the Hyperstack console, create a GPU VM (any NVIDIA flavor, an Ubuntu + CUDA image),
   attach your SSH key, and allow inbound TCP 22. Copy its public IP.
2. From this folder (Git Bash on Windows):

```bash
export HS_HOST=<vm-ip> HS_USER=ubuntu HS_KEY=~/.ssh/id_ed25519
bash hyperstack/remote.sh all --method lora     # push code + .env, set up, start training in tmux
bash hyperstack/remote.sh logs                  # follow progress (Ctrl-C detaches only the tail)
bash hyperstack/remote.sh pull                  # bring output/<run> back here
```
Then evaluate or serve the pulled run locally (below). **Hibernate or delete the VM afterwards;** it bills hourly.

## 4. Evaluate

```bash
python -m sentinel.evaluate --run output/<run> --split holdout
```
Reports PR-AUC, ROC-AUC, ECE, and precision/recall/FPR at the REVIEW and BLOCK thresholds, plus reason-code F1.

## 5. Inference

```bash
python -m sentinel.infer --run output/<run> sample --split holdout --n 5 [--fraud-only]
python -m sentinel.infer --run output/<run> prompt "$(cat my_prompt.txt)"
python -m sentinel.infer --run output/<run> file txns.jsonl     # {"prompt": ...} or structured fields
python -m sentinel.infer --run output/<run> latency --n 50
python -m sentinel.infer --run output/<run> serve --port 8000
curl -s localhost:8000/score -d '{"transaction_id":"TX1","amount":75000,"hour":3,"new_device":true,"fails_24h":4}'
```
Structured input fields are listed in `sentinel.infer.build_prompt`.
