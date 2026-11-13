"""Score transactions with a trained run and return SENTINEL JSON decisions.

    # score N sample rows from a split (shows label next to prediction)
    python -m sentinel.infer --run output/<run> sample --split holdout --n 5
    # score a raw fmt-v1 prompt
    python -m sentinel.infer --run output/<run> prompt "[TXN] id=... [TASK] ..."
    # score a JSON / JSONL file of transactions ({"prompt": ...} or structured fields)
    python -m sentinel.infer --run output/<run> file txns.jsonl
    # HTTP:  POST /score  {"prompt": ...} | {"transactions": [...]}   GET /health
    python -m sentinel.infer --run output/<run> serve --port 8000
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from sentinel import REASON_CODES

logger = logging.getLogger(__name__)

REASON_TEXT = {
    "R01_NEW_DEVICE": "new device",
    "R02_GEO_MISMATCH": "location differs from merchant/home city",
    "R03_ODD_HOUR": "unusual hour",
    "R04_FAILED_ATTEMPTS": "recent failed attempts",
    "R05_AMOUNT_SPIKE": "amount far above the customer's average",
    "R06_NEW_ACCOUNT": "new account",
    "R07_VELOCITY": "high transaction velocity",
    "R08_HIGH_RISK_MCC": "high-risk merchant category",
}
DECISIONS = ("ALLOW", "REVIEW", "BLOCK")


# ------------------------------------------------------------------ prompt
def build_prompt(t: Dict) -> str:
    """Serialise a structured transaction into the fmt-v1 prompt used in training.

    Keys (all optional except amount): transaction_id, amount, mcc, channel, hour, weekday,
    customer_age, account_age_days, avg_90d, txns_30d, fails_24h, new_device (bool),
    city, merchant_city, new_city (bool), dist_km (bucket str), amt_ratio, txns_1h,
    amount_24h, device_users_7d, history: [{days_ago, channel, amount, mcc, city}, ...]
    """
    yn = lambda v: "yes" if v else "no"  # noqa: E731
    num = lambda k, fmt: "unk" if t.get(k) is None else format(float(t[k]), fmt)  # noqa: E731
    hist = t.get("history") or []
    hist_s = " | ".join(
        f"-{h.get('days_ago', 0)}d:{h.get('channel', 'UPI')} {float(h.get('amount', 0)):.1f} "
        f"{h.get('mcc', 'unknown')} {h.get('city', 'unknown')}" for h in hist[:20]
    )
    return (
        f"[TXN] id={t.get('transaction_id', 'TX_NEW')} amt={float(t['amount']):.1f} "
        f"mcc={t.get('mcc', 'unknown')} channel={t.get('channel', 'UPI')} "
        f"time={int(t.get('hour', 12)):02d}h {t.get('weekday', 'Mon')}\n"
        f"[CUST] age={t.get('customer_age', 35)} acct_age_days={t.get('account_age_days', 365)} "
        f"avg_90d={num('avg_90d', '.2f')} txns_30d={t.get('txns_30d', 0)} "
        f"fails_24h={t.get('fails_24h', 0)}\n"
        f"[CTX] new_device={yn(t.get('new_device'))} city={t.get('city', 'unknown')} "
        f"merchant_city={t.get('merchant_city', 'unknown')} new_city={yn(t.get('new_city'))} "
        f"dist_km={t.get('dist_km', '0-50')} amt_ratio={num('amt_ratio', '.1f')} "
        f"txns_1h={t.get('txns_1h', 0)} amount_24h={float(t.get('amount_24h', 0)):.2f} "
        f"device_users_7d={t.get('device_users_7d', 0)}\n"
        f"[HIST] {hist_s if hist_s else 'none observed'}\n"
        "[TASK] Assess fraud risk using only the observed context."
    )


# ------------------------------------------------------------------ decision
def decide(prob: float, review_thr: float, block_thr: float) -> str:
    if prob >= block_thr:
        return "BLOCK"
    if prob >= review_thr:
        return "REVIEW"
    return "ALLOW"


def explain(decision: str, reasons: List[str]) -> str:
    if not reasons:
        return ("No strong risk signals observed." if decision == "ALLOW"
                else "Risk score elevated by the combination of transaction context and history.")
    parts = [REASON_TEXT.get(r, r) for r in reasons]
    lead = {"ALLOW": "Allowed despite", "REVIEW": "Sent to review:", "BLOCK": "Blocked:"}[decision]
    return f"{lead} {', '.join(parts)}."


def to_response(prob: float, reason_logits: np.ndarray, cal: Dict,
                transaction_id: Optional[str] = None) -> Dict:
    decision = decide(prob, cal["review_threshold"], cal["block_threshold"])
    reason_p = 1 / (1 + np.exp(-reason_logits))
    reasons = [c for c, p in zip(REASON_CODES, reason_p) if p >= 0.5]
    return {
        "transaction_id": transaction_id,
        "fraud_probability": round(float(prob), 6),
        "decision": decision,
        "confidence": round(float(max(prob, 1 - prob)), 4),
        "reason_codes": reasons,
        "explanation": explain(decision, reasons),
    }


RESPONSE_SCHEMA_KEYS = {"transaction_id", "fraud_probability", "decision", "confidence",
                        "reason_codes", "explanation"}


def validate_response(r: Dict) -> bool:
    return (set(r) == RESPONSE_SCHEMA_KEYS
            and 0.0 <= r["fraud_probability"] <= 1.0
            and r["decision"] in DECISIONS
            and 0.0 <= r["confidence"] <= 1.0
            and all(c in REASON_CODES for c in r["reason_codes"])
            and isinstance(r["explanation"], str) and r["explanation"] != "")


# ------------------------------------------------------------------ scorer
class SentinelScorer:
    def __init__(self, run_dir: str, review_threshold: Optional[float] = None,
                 block_threshold: Optional[float] = None):
        from sentinel.model import load_trained

        self.run = Path(run_dir)
        self.model, self.tok, self.meta = load_trained(run_dir)
        self.cal = json.loads((self.run / "calibration.json").read_text())
        if review_threshold is not None:
            self.cal["review_threshold"] = review_threshold
        if block_threshold is not None:
            self.cal["block_threshold"] = block_threshold

    def score(self, items: List[Union[str, Dict]], batch_size: int = 32) -> List[Dict]:
        from sentinel.evaluate import apply_platt
        from sentinel.model import predict_logits

        prompts, ids = [], []
        for it in items:
            if isinstance(it, str):
                prompts.append(it); ids.append(None)
            else:
                prompts.append(it["prompt"] if "prompt" in it else build_prompt(it))
                ids.append(it.get("transaction_id"))
        logits = predict_logits(self.model, self.tok, prompts, self.meta["max_length"], batch_size).numpy()
        probs = apply_platt(logits[:, 0], self.cal)
        return [to_response(p, lg[1:], self.cal, i) for p, lg, i in zip(probs, logits, ids)]

    def latency_ms(self, prompt: str, n: int = 20) -> Dict[str, float]:
        self.score([prompt])  # warm-up
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            self.score([prompt])
            ts.append((time.perf_counter() - t0) * 1000)
        return {"p50_ms": float(np.percentile(ts, 50)), "p99_ms": float(np.percentile(ts, 99))}


# ------------------------------------------------------------------ server
def serve(scorer: SentinelScorer, host: str, port: int):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading

    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok", "run": scorer.run.name,
                                 "thresholds": {k: scorer.cal[k] for k in ("review_threshold", "block_threshold")}})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/score":
                return self._send(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                items = body.get("transactions") or [body]
                t0 = time.perf_counter()
                with lock:
                    res = scorer.score(items)
                ms = (time.perf_counter() - t0) * 1000
                self._send(200, {"results": res, "latency_ms": round(ms, 2)})
            except (KeyError, ValueError, TypeError) as e:
                self._send(400, {"error": str(e)})

        def log_message(self, fmt, *args):
            logger.info("%s %s", self.address_string(), fmt % args)

    srv = ThreadingHTTPServer((host, port), H)
    logger.info("Serving on http://%s:%d  (POST /score, GET /health)", host, port)
    srv.serve_forever()


def main(argv=None):
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description="SENTINEL inference")
    p.add_argument("--run", required=True)
    p.add_argument("--review-threshold", type=float)
    p.add_argument("--block-threshold", type=float)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample"); s.add_argument("--split", default="holdout"); s.add_argument("--n", type=int, default=5)
    s.add_argument("--fraud-only", action="store_true"); s.add_argument("--data-dir", default="data/sft")
    s = sub.add_parser("prompt"); s.add_argument("text")
    s = sub.add_parser("file"); s.add_argument("path")
    s = sub.add_parser("latency"); s.add_argument("--n", type=int, default=50); s.add_argument("--data-dir", default="data/sft")
    s = sub.add_parser("serve"); s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8000)
    a = p.parse_args(argv)

    scorer = SentinelScorer(a.run, a.review_threshold, a.block_threshold)
    if a.cmd == "serve":
        return serve(scorer, a.host, a.port)
    if a.cmd == "prompt":
        print(json.dumps(scorer.score([a.text])[0], indent=2))
    elif a.cmd == "file":
        text = Path(a.path).read_text(encoding="utf-8").strip()
        items = json.loads(text) if text.startswith("[") else [json.loads(l) for l in text.splitlines() if l.strip()]
        for r in scorer.score(items):
            print(json.dumps(r))
    elif a.cmd == "sample":
        from sentinel.data import load_split

        rows = load_split(a.split, Path(a.data_dir))
        if a.fraud_only:
            rows = [r for r in rows if r["fraud_label"] == 1]
        rows = rows[: a.n]
        for row, res in zip(rows, scorer.score([{"prompt": r["prompt"], "transaction_id": r["transaction_id"]} for r in rows])):
            print(json.dumps({**res, "actual_fraud_label": row["fraud_label"]}))
    elif a.cmd == "latency":
        from sentinel.data import load_split

        prompt = load_split("holdout", Path(a.data_dir), limit=1)[0]["prompt"]
        print(json.dumps(scorer.latency_ms(prompt, a.n), indent=2))


if __name__ == "__main__":
    main()
