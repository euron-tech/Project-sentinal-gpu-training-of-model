#!/usr/bin/env bash
# Run ON the VM. Serves a trained run on the GPU at 0.0.0.0:<port> in tmux session "serve".
#   bash hyperstack/serve.sh                       # latest run in output/, port 8000
#   bash hyperstack/serve.sh output/<run> 8000
# Requires SENTINEL_API_KEY (in .env or the environment); one is generated into .env if missing.
# Testers open  http://<VM_IP>:<port>/   (needs a firewall rule for inbound TCP <port>).
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${1:-$(ls -1dt output/*/ 2>/dev/null | grep -v checkpoints | head -1)}"
PORT="${2:-8000}"
[ -n "$RUN" ] && [ -f "$RUN/calibration.json" ] || { echo "No finished run found (need output/<run>/calibration.json)"; exit 1; }

touch .env
if ! grep -q '^SENTINEL_API_KEY=' .env && [ -z "${SENTINEL_API_KEY:-}" ]; then
  echo "SENTINEL_API_KEY=$(python3 -c 'import secrets;print(secrets.token_urlsafe(24))')" >> .env
  echo "Generated SENTINEL_API_KEY in .env"
fi
KEY="${SENTINEL_API_KEY:-$(grep '^SENTINEL_API_KEY=' .env | tail -1 | cut -d= -f2-)}"

tmux kill-session -t serve 2>/dev/null || true
tmux new-session -d -s serve \
  "source .venv/bin/activate && SENTINEL_API_KEY='$KEY' python -m sentinel.infer --run '${RUN%/}' serve --host 0.0.0.0 --port $PORT 2>&1 | tee output/serve.log"

IP="${PUBLIC_IP:-<VM_PUBLIC_IP>}"
echo "Serving ${RUN%/} on GPU (tmux session 'serve', log output/serve.log)"
echo "  Test page : http://$IP:$PORT/"
echo "  Health    : curl http://$IP:$PORT/health"
echo "  API key   : $KEY   (share only with testers; header X-API-Key)"
