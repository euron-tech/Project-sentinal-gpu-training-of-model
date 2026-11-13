#!/usr/bin/env bash
# Run ON the VM. Serves a trained run on the GPU at 0.0.0.0:<port> in tmux session "serve".
#   bash hyperstack/serve.sh                       # newest FINISHED run in output/, port 8000
#   bash hyperstack/serve.sh output/<run> 8000
# Requires SENTINEL_API_KEY (in .env or the environment); one is generated into .env if missing.
# Testers open  http://<VM_IP>:<port>/   (needs a firewall rule for inbound TCP <port>).
# Stop: tmux kill-session -t serve
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x .venv/bin/python ] || { echo "Run: bash hyperstack/setup_vm.sh first"; exit 1; }

latest_finished() {
  for d in $(ls -1dt output/*/ 2>/dev/null); do
    [ -f "${d}calibration.json" ] && [ -f "${d}adapter/adapter_config.json" ] && { echo "${d%/}"; return; }
  done
}
RUN="${1:-$(latest_finished || true)}"
RUN="${RUN%/}"
PORT="${2:-8000}"
[ -n "$RUN" ] && [ -f "$RUN/calibration.json" ] || { echo "No finished run found (need output/<run>/calibration.json). Train first: bash hyperstack/train.sh"; exit 1; }

touch .env
if ! grep -q '^SENTINEL_API_KEY=' .env && [ -z "${SENTINEL_API_KEY:-}" ]; then
  echo "SENTINEL_API_KEY=$(.venv/bin/python -c 'import secrets;print(secrets.token_urlsafe(24))')" >> .env
  echo "Generated SENTINEL_API_KEY in .env"
fi
KEY="${SENTINEL_API_KEY:-$(grep '^SENTINEL_API_KEY=' .env | tail -1 | cut -d= -f2- | tr -d '\r')}"

tmux kill-session -t serve 2>/dev/null || true
mkdir -p output
: > output/serve.log
tmux new-session -d -s serve "bash hyperstack/_tmux_run.sh serve '$RUN' $PORT '$KEY'"

echo -n "Starting server (loading model on GPU)"
for _ in $(seq 1 120); do
  if curl -fs "http://localhost:$PORT/health" >/dev/null 2>&1; then break; fi
  echo -n "."; sleep 2
done
echo
if ! curl -fs "http://localhost:$PORT/health" >/dev/null 2>&1; then
  echo "Server did not come up. Last log lines:"; tail -n 30 output/serve.log; exit 1
fi

IP="${PUBLIC_IP:-<VM_PUBLIC_IP>}"
echo "Serving $RUN  ->  $(curl -s http://localhost:$PORT/health)"
echo "  Test page : http://$IP:$PORT/"
echo "  Health    : curl http://$IP:$PORT/health"
echo "  API key   : $KEY   (share only with testers; header X-API-Key)"
echo "  Auto-restarts if it crashes. Stop with: tmux kill-session -t serve"
