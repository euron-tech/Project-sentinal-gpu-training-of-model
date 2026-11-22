# SENTINEL: real-time fraud scoring. Models on Hyperstack GPU → Amazon Kinesis → decision → action, dashboard on AWS (ap-south-1)

```
  Hyperstack GPU VM (all models)          AWS ap-south-1 / Mumbai (all Terraform, own VPC 10.40.0.0/16)
  ┌────────────────────────────┐          ┌────────────────────────────────────────────────────────────────────┐
  │ LoRA    API :8000          │          │ Kinesis Data Streams          ALB  /        → ui     :8501           │
  │ XGBoost API :8001          │◀── HTTP ─┤  sentinel-prod-transactions        /live/*  → stream :8003           │
  │ QLoRA   API :8002          │  + key   │  sentinel-prod-decisions       ECS Fargate: ONE task, 2 containers  │
  │ hyperstack/pipeline.sh     │          │  (24 h retention, 1 shard)    ECR · S3 · SSM · CloudWatch · SNS      │
  └────────────────────────────┘          └────────────────────────────────────────────────────────────────────┘
  GitHub ── push to main ──▶ Actions: tests → build + push to ECR → terraform apply → Fargate rollout (OIDC, no AWS keys)
```

* **All models run on the Hyperstack GPU VM**: LoRA, QLoRA and XGBoost are trained there and served from there.
* **AWS (ap-south-1) only hosts the result**: the dashboard, the analytics, and the stream scorer that calls the Hyperstack
  APIs. It is one ECS Fargate service and there are no EC2 instances.
* **Streaming is Amazon Kinesis Data Streams**, in the same AWS account and region. Terraform creates two
  encrypted streams (one shard each) with the rest of the stack, and destroys them. No third-party account or key is needed.
* **Traffic flows only when you press Start** in the dashboard (**Live Flow** tab). Every transaction is scored by XGBoost
  (champion), with LoRA and QLoRA scoring in shadow. Each decision triggers an action:
  * `ALLOW`: non-fraud, approved, no further action.
  * `REVIEW`: OTP step-up.
  * `BLOCK`: card frozen, alert.

  Every record on the Kinesis stream `sentinel-prod-decisions` carries its action (downstream systems read it there),
  and BLOCKs also go to SNS.

---

## 0. Isolation: what this touches in ap-south-1 (and what it never touches)

The account already runs other services in ap-south-1. This project is built so it cannot affect them.

| Guarantee | How |
|---|---|
| Own network | New VPC `10.40.0.0/16`, own subnets, IGW, ALB and security groups. Nothing is shared with other VPCs, no peering. |
| Own names | Every resource is named `sentinel-prod-*` (ECS cluster, ALB, ECR `sentinel-prod/*`, IAM roles, log groups, SSM `/sentinel-prod/*`, buckets). |
| Own tags | Every resource is tagged `Project=sentinel`, `ManagedBy=terraform`. |
| Own state | Terraform state lives in its own bucket `sentinel-tfstate-<account>-ap-south-1`. `destroy` can only remove what is in that state. |
| Checked first | `python scripts/ops.py check` (read-only) verifies that no existing resource uses these names, that a VPC slot is free (quota 5 per region), and that the models answer. `up` refuses to start if it fails. |
| Clean teardown | `ops.py down` destroys exactly this stack (ECR images and buckets included). `ops.py audit` lists anything still tagged `Project=sentinel`. |

Account-wide items (IAM is global, not per region):
* The role `sentinel-github-deploy`.
* The GitHub OIDC provider `token.actions.githubusercontent.com`, created only if the account has none and removed by
  `down --all` only if this project created it.

The deploy role can only be assumed by GitHub Actions runs from this repository's `main` branch or `production`
environment.

---

## 1. Configuration checklist: everything needed for one-click deployment

| # | Setting | Where it comes from | Goes into |
|---|---|---|---|
| 1 | `TF_VAR_hyperstack_host` | Public IP of the Hyperstack VM, printed by `pipeline.sh` (`output/aws.env`) | `.env` on your PC |
| 2 | `TF_VAR_model_api_key` | Generated on the VM, in `output/aws.env` (same as `SENTINEL_API_KEY` in the VM's `.env`) | `.env` on your PC |
| 3 | `SUPABASE_URL`, `SUPABASE_KEY` | Supabase project holding `sentinel-data/training/sft/*` (the training data) | `.env` on the **VM** (and the PC, optional) |
| 4 | AWS credentials | `aws configure` (IAM user with admin rights) or `aws login` | your PC, once |
| 5 | GitHub CLI login | `gh auth login` | your PC, once |
| 6 | *(optional)* `TF_VAR_alert_email` | Your email for CloudWatch alarms (confirm the link AWS sends) | `.env` on your PC |
| 7 | *(optional)* `AWS_REGION` | Default `ap-south-1`. Only set it to deploy somewhere else. | `.env` on your PC |

You do **not** create these, because Terraform does:
* The Kinesis streams and their IAM permissions.
* The control key for the Start/Stop buttons.
* The secrets in SSM, and the GitHub variables and secrets for CI/CD.

`.env` is git-ignored. Start it from the template: `copy .env.example .env`.

**Tools on your PC:** Python 3.10+, Git, AWS CLI v2, Terraform ≥ 1.9, Docker Desktop (running), GitHub CLI.

---

## 2. End-to-end runbook (run in this order)

### Step 1: Start the Hyperstack GPU VM
1. In the Hyperstack console, **start/restore** the VM, or create one: a single NVIDIA GPU (≥ 16 GB), an Ubuntu image
   *with NVIDIA drivers/CUDA*, your SSH key, and a public IP.
2. Add firewall rules for inbound **TCP 22, 8000, 8001, 8002** from `0.0.0.0/0`. AWS Fargate has no fixed IP; the API
   key protects the model ports.
3. Note the **public IP**. It can change after a hibernate/restore.

### Step 2: Train and serve all models on the VM (one command, unattended)
```bash
ssh ubuntu@<VM_IP>
git clone https://github.com/euron-tech/Project-sentinal-gpu-training-of-model.git sentinel   # first time only
cd sentinel && git pull
cp -n .env.example .env && nano .env        # first time only: SUPABASE_URL + SUPABASE_KEY
bash hyperstack/pipeline.sh                 # runs in tmux; safe to close SSH
tail -f output/pipeline.log                 # wait for "PIPELINE DONE" (Ctrl-C only stops tail)
cat output/aws.env                          # the 2 lines for your PC
```
`pipeline.sh` runs these six steps by itself:
1. Setup: venv, CUDA torch, dependencies, data, Qwen weights.
2. Health check.
3. Train **LoRA**.
4. Train **QLoRA**.
5. Train **XGBoost**.
6. Serve all three (LoRA :8000, XGBoost :8001, QLoRA :8002, auto-restart on crash), then write `output/aws.env`.

| Variant | Command |
|---|---|
| Full training (~20–30 min) | `bash hyperstack/pipeline.sh` |
| Short demo training (~5 min) | `QUICK=1 bash hyperstack/pipeline.sh` |
| VM was restarted, models already trained, just serve | `SKIP_TRAIN=1 bash hyperstack/pipeline.sh` |
| Health of GPU, data, runs, APIs | `bash hyperstack/check.sh` |

### Step 3: Configure your PC (repo root, PowerShell)
```powershell
git pull
copy .env.example .env      # first time only, then edit .env:
#   TF_VAR_hyperstack_host=<VM_IP>      TF_VAR_model_api_key=<from output/aws.env>
python scripts/ops.py check  # read-only; must end with "ready to deploy"
```

### Step 4: Deploy everything (one click)
```powershell
python scripts/ops.py up
```
It runs these steps by itself (about 15–20 minutes the first time):
1. **check**: settings, models reachable, isolation in ap-south-1.
2. **bootstrap**: state bucket, GitHub OIDC deploy role, GitHub variables and secrets.
3. **ECR**: create the image repositories.
4. **Images**: build and push the `ui` and `stream` images.
5. **Dataset**: copy the labelled dataset to S3.
6. **Terraform apply**: Kinesis streams, VPC, ALB, ECS Fargate, CloudWatch, SNS.
7. **Wait** until the service is healthy, then **print the URLs**.

`python scripts/ops.py status` shows the URLs and health again at any time.

### Step 5: Use it
1. Open the **dashboard URL** that was printed, go to **Live Flow**, and press **Start auto traffic**.
2. Set a rate, and use **Inject attack** to send a fraud-ring wave.
3. Watch the animated flow (Kinesis → scorer → models → decision → action) and the live decision feed. The flow page alone
   is at `<url>/live/flow`.
4. **Analytics** shows:
   * Who sends risky traffic: device, customer, merchant, IP, fraud ring.
   * Breakdowns by category, channel, hour and city.
   * Model recall, precision and latency.
   * CPU, memory and disk of the Fargate task.
5. **CloudWatch** has the dashboard `sentinel-prod`, the logs `/ecs/sentinel-prod/{ui,stream}`, and alarms that go to
   SNS `sentinel-prod-alerts`.
6. Press **Stop** to end the traffic.

### Step 6: Change code → automatic deployment (CI/CD)
`git push` to `main` runs the following, with no AWS keys stored in GitHub:
1. **tests**: stream, XGBoost, dashboard, LLM, and `terraform validate`.
2. **build + push** the images to ECR.
3. **terraform apply**.
4. **Fargate rollout**.

| Workflow | Trigger | Does |
|---|---|---|
| `ci.yml` | every PR / push | all tests + `terraform fmt` / `validate` |
| `deploy.yml` | push to `main`, or *Run workflow* | ci → OIDC role → `python scripts/ops.py deploy --tag <sha>` |
| `destroy.yml` | manual, type `sentinel-prod` | `python scripts/ops.py down -y` |

### Step 7: The VM was restarted and its IP changed
```bash
# VM
cd sentinel && SKIP_TRAIN=1 bash hyperstack/pipeline.sh && tail -f output/pipeline.log
```
```powershell
# PC: put the new IP in .env (TF_VAR_hyperstack_host), then
python scripts/ops.py deploy      # or: gh variable set HYPERSTACK_HOST --body <new IP>, then re-run deploy.yml
```

### Step 8: Shut down (stops all costs)
```powershell
python scripts/ops.py down        # type sentinel-prod; removes only this project's AWS resources (Kinesis included)
python scripts/ops.py audit       # confirms nothing tagged Project=sentinel is left
python scripts/ops.py down --all  # also removes the state bucket + GitHub deploy role (full clean-up)
```
On the VM, run `tmux kill-server`, then **hibernate the VM** in the Hyperstack console. It bills hourly.

---

## 3. Data and storage

| Where | What | Why |
|---|---|---|
| Kinesis Data Streams | `sentinel-prod-transactions` (24 h), `sentinel-prod-decisions` with the action (24 h) | Durable record of every transaction, decision and action. |
| SQLite in the stream container | Scored decisions for the dashboard queries | Fast analytics. It is rebuilt from the last 24 h of `sentinel-prod-decisions` when the task restarts. |
| S3 `sentinel-prod-data-<account>` | The labelled holdout dataset | The producer replays it, mixed with synthetic fraud rings. |
| SSM `/sentinel-prod/*` (encrypted) | Model key, control key | Injected into the containers; never in code or GitHub. |
| CloudWatch | Logs, Container Insights, app metrics, alarms | Monitoring. |

If no Hyperstack model answers, a transaction is held for **REVIEW** (fail closed).

## 4. Cost (approximate, if left running)

| Item | Cost |
|---|---|
| Fargate 1 vCPU / 2 GB | ~$35 / month |
| ALB | ~$18 / month |
| CloudWatch | a few $ / month |
| Kinesis: 2 streams × 1 shard, 24 h retention | ~$25 / month |

The Hyperstack VM is billed separately by the hour. `ops.py down` and hibernating the VM stop everything.

## 5. Troubleshooting the deployment

| Symptom | Fix |
|---|---|
| `check`: WARN Hyperstack not reachable | Is `pipeline.sh` done? Is the firewall open for 8000–8002? Is the IP current? `curl http://<VM_IP>:8001/health` |
| `check`: FAIL VPC quota | ap-south-1 allows 5 VPCs by default. Request an increase in Service Quotas (VPCs per Region). |
| `check`: FAIL names already exist | A previous deployment's state was lost. Run `ops.py audit`, then remove the listed `sentinel-prod` items, or set `ENVIRONMENT=prod2` in `.env`. |
| Dashboard says models offline | The Hyperstack IP changed (Step 7), or the API key in `.env` differs from the VM's `output/aws.env` |
| Every decision is REVIEW | No model reachable (fail-closed). Same fixes as above. |
| `deploy.yml` deploy job skipped | `ops.py up` has not run yet; it sets the GitHub variable `AWS_ROLE_ARN` |

## 6. Run the pipeline locally (no AWS)
```bash
docker compose up -d --build        # local Kinesis emulator + stream (producer/scorer) + XGBoost + dashboard
start http://localhost:8501         # Live Flow -> Start auto traffic
cd stream && python -m pytest -q
```

---

## Development and training (Hyperstack GPU, local)

## Showcase runbook: 3 models on the GPU, dashboard on your PC (run line by line)

```
 Hyperstack GPU VM                                  Your PC
 ┌──────────────────────────────┐                  ┌──────────────────────────────┐
 │ LoRA     API :8000  (GPU)    │◀────── HTTP ─────│ Dashboard http://localhost:8501│
 │ XGBoost  API :8001  (GPU)    │◀─────────────────│ Overview · LoRA vs QLoRA ·   │
 │ QLoRA    API :8002  (GPU)    │◀─────────────────│ Benchmark · Live · Batch ·   │
 └──────────────────────────────┘   one API key    │ Decision (GO / NO-GO)        │
                                                   └──────────────────────────────┘
```
XGBoost is the model in production today (baseline). LoRA and QLoRA are the new candidates.
Hyperstack firewall: open inbound TCP **22, 8000, 8001, 8002**.

**1. On the VM** (`ssh ubuntu@<VM_IP>`):
```bash
cd sentinel
git pull
bash hyperstack/setup_vm.sh                      # safe to re-run; now also installs XGBoost
bash hyperstack/check.sh --full                  # must end with ALL GOOD

bash hyperstack/train.sh                         # LoRA (~8 min)
tail -f output/last_train.log                    # wait for "Artifacts in output/...", then Ctrl-C
bash hyperstack/train.sh --method qlora          # QLoRA (~8 min)
tail -f output/last_train.log                    # wait again, then Ctrl-C
bash hyperstack/xgb.sh train                     # XGBoost on the GPU (~1 min) -> output/xgb

PUBLIC_IP=<VM_IP> bash hyperstack/serve_all.sh   # starts LoRA :8000, XGBoost :8001, QLoRA :8002
bash hyperstack/check.sh                         # all three lines say "API up"
```
`serve_all.sh` ends by printing the exact line to run on your PC (with the API key filled in).

**2. On your PC** (PowerShell, in `D:\finetuning`):
```powershell
git pull
& ui\run_service.ps1 -VmIp <VM_IP> -Key <key> -Rebuild   # -Rebuild only the first time after a pull
start http://localhost:8501                              # three green dots in the sidebar
```
No Docker? Run it directly instead:
```powershell
$env:LORA_URL="http://<VM_IP>:8000"; $env:QLORA_URL="http://<VM_IP>:8002"; $env:XGB_URL="http://<VM_IP>:8001"
$env:LLM_API_KEY="<key>"; $env:XGB_API_KEY="<key>"
streamlit run ui/app.py
```

**3. The demo, tab by tab:** Overview (KPIs for all three) → LoRA vs QLoRA (training time, loss, thresholds) →
Benchmark (PR/ROC curves, Δ vs XGBoost) → Live Test (score a scenario on all three; do **at least 5 calls** so the
latency check fills in) → Batch Test (200 labelled holdout rows on all three) → **Decision**: GO (deploy the best
candidate) or NO-GO (keep XGBoost), with the checklist below it. Click **Download decision report** for the meeting notes.
The targets (recall ≥ 85 %, precision at BLOCK ≥ 70 %, PR-AUC ≥ 0.80, uplift ≥ +0.03, no worse false-positive rate,
p99 ≤ 250 ms) can be changed in the sidebar under *Acceptance targets*.

**4. Optional: copy the trained models to your PC** (Git Bash or PowerShell):
```bash
scp -r ubuntu@<VM_IP>:sentinel/output ./output-from-vm   # LoRA + QLoRA run folders and output/xgb
```

**5. When done:** `tmux kill-server` on the VM, then **hibernate the VM**.

---

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

**Console:** single-GPU NVIDIA VM (any card with ≥ 16 GB works; training needs only about 2.5 GB), an Ubuntu image with
NVIDIA drivers/CUDA, your SSH key, public IP. Open inbound TCP **22** and **8000**.

```bash
ssh ubuntu@<VM_IP>
git clone https://github.com/euron-tech/Project-sentinal-gpu-training-of-model.git sentinel && cd sentinel
cp .env.example .env && nano .env                # SUPABASE_URL, SUPABASE_KEY
bash hyperstack/setup_vm.sh                      # apt, venv, CUDA torch (auto-fallback), deps, data, Qwen weights -> "VM ready."
bash hyperstack/check.sh --full                  # GPU, torch, data, model, 15 tests -> "ALL GOOD"

bash hyperstack/train.sh                         # LoRA, full run: ~5 min train + ~3 min eval on a 4090-class GPU
bash hyperstack/train.sh --method qlora          # 4-bit QLoRA variant
bash hyperstack/train.sh --max-steps 30 --max-eval-rows 2000   # 1-minute demo run
tail -f output/last_train.log                    # done at "Artifacts in output/<run>"

source .venv/bin/activate
RUN=$(ls -1dt output/*/ | head -1)
python -m sentinel.evaluate --run $RUN --split holdout
python -m sentinel.infer --run $RUN sample --split holdout --n 5 --fraud-only

PUBLIC_IP=<VM_IP> bash hyperstack/serve.sh       # GPU API on :8000 (auto-restarts); waits until healthy; prints URL + key
bash hyperstack/check.sh                         # any time: what is running, latest run metrics
```
Check from your PC: `curl http://<VM_IP>:8000/health` (should show `"device": "cuda:0"`).
`train.sh` will not start a second run while one is active (`FORCE=1` replaces it).
To update: `git pull && bash hyperstack/setup_vm.sh`, then retrain or re-serve. **Hibernate the VM when done.**

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

Always-on (Docker, auto-restart); settings persist in `ui/.env`:
```powershell
& ui\run_service.ps1 -LlmUrl http://<VM_IP>:8000 -LlmKey <key from serve.sh>     # http://localhost:8501
& ui\run_service.ps1 -XgbUrl https://<id>.ngrok-free.app                        # optional: XGBoost via ngrok
```
Or without Docker:
```bash
# PowerShell: $env:LLM_URL="http://<VM_IP>:8000"; $env:LLM_API_KEY="<from serve.sh>"
#             $env:XGB_URL="http://localhost:8001"; $env:XGB_API_KEY="<xgboost_benchmark/.env>"
streamlit run ui/app.py                          # URLs/keys can also be changed live in the sidebar
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

## Class-day checklist
1. Your PC: `docker ps` shows `sentinel-xgb` and `sentinel-ui` as *healthy* (Docker Desktop: *Start when you sign in*).
2. VM: `bash hyperstack/check.sh --full` prints **ALL GOOD**.
3. VM: `bash hyperstack/train.sh` → `tail -f output/last_train.log` → `PUBLIC_IP=<VM_IP> bash hyperstack/serve.sh`.
4. PC: `& ui\run_service.ps1 -LlmUrl http://<VM_IP>:8000 -LlmKey <key>` → open http://localhost:8501. Both dots are green.
5. Sharing: `ngrok http 8501` for the dashboard (the XGBoost API needs its own `ngrok http 8001` only for remote dashboards).

## Troubleshooting
| Symptom | Fix |
|---|---|
| Dashboard shows LLM offline | VM firewall needs TCP 8000; `tmux ls` on the VM should list `serve`; check the key |
| XGBoost offline via ngrok | Is `uvicorn` still running? Is the ngrok URL current (it changes on restart)? Is `XGB_API_KEY` the same? |
| 401 | Wrong or missing API key in the sidebar |
| Dashboard shows QLoRA or XGBoost offline | VM firewall needs TCP 8002 / 8001; `tmux ls` should list `serve-8002` / `xgb`; `serve_all.sh` skips a model that has no finished run |
| Decision says *pending* | A model is offline, or fewer than 5 Live Test calls have been made (latency check) |
| `setup_vm.sh`: `cuda False` | Image driver too old. Use a newer image, or `TORCH_INDEX=https://download.pytorch.org/whl/cu128` |
| CUDA OOM while training | Gradient checkpointing is on by default (~2.5 GB). Still OOM: `--batch-size 8 --grad-accum 2` |
| `train.sh` says a run is in progress | `tail -f output/last_train.log`, or `FORCE=1 bash hyperstack/train.sh ...` to replace it |
| `serve.sh`: server did not come up | It prints the log tail; usually no finished run yet, or port 8000 already in use (`tmux kill-session -t serve`) |

Plain HTTP with shared keys is for testing only. Put the APIs behind HTTPS before using real data.
