"""Stream service API (port 8003). Everything is under /live so one load balancer rule can route it.

    uvicorn sentinel_stream.api:app --host 0.0.0.0 --port 8003
  GET  /live/health
  GET  /live/  (/live/flow)        animated end-to-end flow page (Kafka -> models -> decision -> action)
  GET  /live/api/flow?since=<id>   pipeline state + events newer than <id> (polled by the flow page)
  GET  /live/api/summary?minutes=60  application analytics (who sends what, decisions, models, reasons)
  GET  /live/api/events?limit=50   latest scored transactions
  GET  /live/api/system            host CPU / memory / disk of the instance this runs on
  POST /live/api/producer          {"action": "start"|"stop", "rate": 2}   auto-mode traffic   needs X-Control-Key
  POST /live/api/inject            {"n": 20, "fraud_type": "account_takeover"}  one attack wave  needs X-Control-Key
Env: KAFKA_BOOTSTRAP (+ KAFKA_API_KEY / KAFKA_API_SECRET for Confluent Cloud; processor runs when set),
     STORE_PATH, CONTROL_API_KEY, AUTO_START (0), plus processor / producer env.
"""

from __future__ import annotations

import hmac
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse

from sentinel_stream.generator import FRAUD_TYPES
from sentinel_stream.processor import Processor
from sentinel_stream.producer import Traffic
from sentinel_stream.store import Store

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("sentinel.api")
FLOW_HTML = (Path(__file__).parent / "flow.html").read_text(encoding="utf-8")

S: Dict = {}


def store() -> Store:
    if "store" not in S:
        S["store"] = Store(os.getenv("STORE_PATH", "/data/sentinel.db"), float(os.getenv("RETENTION_HOURS", "72")))
    return S["store"]


@asynccontextmanager
async def lifespan(_app):
    if os.getenv("KAFKA_BOOTSTRAP"):
        S["processor"] = Processor(store())
        S["processor"].start()
        S["traffic"] = Traffic()
        if os.getenv("AUTO_START", "0") == "1":
            S["traffic"].start()
    else:
        logger.warning("KAFKA_BOOTSTRAP not set - API only, no stream processing")
    yield
    if S.get("traffic"):
        S["traffic"].stop()
    if S.get("processor"):
        S["processor"].stop()


app = FastAPI(title="SENTINEL stream", version="1.0", docs_url="/live/docs", openapi_url="/live/openapi.json",
              lifespan=lifespan)


def _pipeline() -> Dict:
    p: Optional[Processor] = S.get("processor")
    t: Optional[Traffic] = S.get("traffic")
    if not p:
        return {"kafka": "disabled", "models": {}, "consumed": 0, "producer": {"running": False}}
    models = {n: {"enabled": c.enabled, "healthy": bool(c.healthy), "status": c.status, "scored": c.scored,
                  "errors": c.errors, "last_ms": c.last_ms, "role": "champion" if n == p.champion else "shadow"}
              for n, c in p.clients.items()}
    return {**p.state, "models": models, "uptime_s": time.time() - p.state["started"],
            "producer": t.status() if t else {"running": False}}


@app.get("/live/health")
def health():
    return {"status": "ok", "kafka": _pipeline().get("kafka"), "events": store().max_id()}


@app.get("/live", response_class=HTMLResponse, include_in_schema=False)
@app.get("/live/", response_class=HTMLResponse, include_in_schema=False)
@app.get("/live/flow", response_class=HTMLResponse, include_in_schema=False)
def flow_page():
    return FLOW_HTML


@app.get("/live/api/flow")
def flow(since: int = Query(0, ge=0)):
    st = store()
    head = st.max_id()
    if since <= 0 or since > head:           # first call: no backlog animation, just the latest few
        since = max(0, head - 8)
    ev = st.recent(60, since)
    keep = ("id", "ts", "customer_id", "merchant_name", "amount", "mcc", "channel", "city", "decision", "probability",
            "action", "reason_codes", "champion", "fraud_label", "fraud_type",
            "xgboost_decision", "lora_decision", "qlora_decision", "xgboost_p", "lora_p", "qlora_p")
    return {"head": head, "pipeline": _pipeline(), "totals": st.totals(),
            "events": [{k: e.get(k) for k in keep} for e in reversed(ev)]}


@app.get("/live/api/summary")
def summary(minutes: int = Query(60, ge=1, le=60 * 24 * 7)):
    return {**store().summary(minutes), "pipeline": _pipeline()}


@app.get("/live/api/events")
def events(limit: int = Query(50, ge=1, le=1000)):
    return {"events": store().recent(limit)}


@app.get("/live/api/system")
def system():
    import psutil

    vm = psutil.virtual_memory()
    du = psutil.disk_usage(os.getenv("STORE_DIR", "/"))
    return {"cpu_percent": psutil.cpu_percent(interval=0.3), "cpus": psutil.cpu_count(),
            "load_avg": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
            "mem_used_mb": round((vm.total - vm.available) / 2**20), "mem_total_mb": round(vm.total / 2**20),
            "mem_percent": vm.percent, "disk_percent": du.percent}


def _control(key: Optional[str]) -> Traffic:
    want = os.getenv("CONTROL_API_KEY", "")
    if not want or not hmac.compare_digest(key or "", want):
        raise HTTPException(401, "missing or invalid X-Control-Key")
    if "traffic" not in S:
        raise HTTPException(503, "Kafka not configured")
    return S["traffic"]


@app.get("/live/api/producer")
def producer_status():
    t: Optional[Traffic] = S.get("traffic")
    return t.status() if t else {"running": False}


@app.post("/live/api/producer")
def producer_control(body: Dict, x_control_key: Optional[str] = Header(default=None)):
    t = _control(x_control_key)
    action = body.get("action")
    if action == "start":
        return t.start(body.get("rate"))
    if action == "stop":
        return t.stop()
    raise HTTPException(400, "action must be start or stop")


@app.post("/live/api/inject")
def inject(body: Dict, x_control_key: Optional[str] = Header(default=None)):
    t = _control(x_control_key)
    if time.time() - S.get("last_inject", 0) < 3:
        raise HTTPException(429, "one attack wave every 3 seconds")
    n = max(1, min(100, int(body.get("n", 20))))
    ftype = body.get("fraud_type")
    if ftype and ftype not in FRAUD_TYPES:
        raise HTTPException(400, f"fraud_type must be one of {FRAUD_TYPES}")
    sent = t.burst(n, ftype)
    S["last_inject"] = time.time()
    return {"sent": sent, "fraud_type": ftype or "mixed"}
