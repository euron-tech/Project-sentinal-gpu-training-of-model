"""SQLite event store for scored transactions + the analytics queries behind the dashboard.

One writer (the processor thread), many readers (API requests). WAL mode keeps reads non-blocking.
Rows older than RETENTION_HOURS are pruned so the file stays small on the instance disk.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

MODELS = ("xgboost", "lora", "qlora")
DECISIONS = ("ALLOW", "REVIEW", "BLOCK")

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id TEXT UNIQUE, ts REAL, scored_ts REAL, source TEXT,
  customer_id TEXT, merchant_id TEXT, merchant_name TEXT, device_id TEXT, ip TEXT, ring_id TEXT,
  amount REAL, mcc TEXT, channel TEXT, city TEXT, merchant_city TEXT, hour INTEGER,
  fraud_label INTEGER, fraud_type TEXT,
  champion TEXT, decision TEXT, probability REAL, action TEXT, reason_codes TEXT, explanation TEXT,
  xgboost_p REAL, xgboost_decision TEXT, xgboost_ms REAL,
  lora_p REAL, lora_decision TEXT, lora_ms REAL,
  qlora_p REAL, qlora_decision TEXT, qlora_ms REAL
);
CREATE INDEX IF NOT EXISTS ix_events_ts ON events(ts);
"""
COLUMNS = ["event_id", "ts", "scored_ts", "source", "customer_id", "merchant_id", "merchant_name", "device_id", "ip",
           "ring_id", "amount", "mcc", "channel", "city", "merchant_city", "hour", "fraud_label", "fraud_type",
           "champion", "decision", "probability", "action", "reason_codes", "explanation"] + \
          [f"{m}_{k}" for m in MODELS for k in ("p", "decision", "ms")]


class Store:
    def __init__(self, path: str = "/data/sentinel.db", retention_hours: float = 72):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.retention_s = retention_hours * 3600
        self._local = threading.local()
        self._lock = threading.Lock()
        self._shared = sqlite3.connect(self.path, check_same_thread=False) if self.path == ":memory:" else None
        c = self._conn()
        c.executescript(SCHEMA)
        if self.path != ":memory:":
            c.execute("PRAGMA journal_mode=WAL")
        self._last_prune = 0.0

    def _conn(self) -> sqlite3.Connection:
        if self._shared is not None:
            return self._shared
        c = getattr(self._local, "c", None)
        if c is None:
            c = self._local.c = sqlite3.connect(self.path, timeout=10)
            c.row_factory = sqlite3.Row
        return c

    def _rows(self, sql: str, args=()) -> List[dict]:
        with self._lock:
            cur = self._conn().execute(sql, args)
            names = [d[0] for d in cur.description]
            return [dict(zip(names, r)) for r in cur.fetchall()]

    # ------------------------------------------------------------- write
    def insert(self, records: List[dict]) -> None:
        if not records:
            return
        rows = []
        for r in records:
            r = dict(r)
            if isinstance(r.get("reason_codes"), list):
                r["reason_codes"] = ",".join(r["reason_codes"])
            rows.append(tuple(r.get(k) for k in COLUMNS))
        sql = f"INSERT OR IGNORE INTO events ({','.join(COLUMNS)}) VALUES ({','.join('?' * len(COLUMNS))})"
        with self._lock:
            c = self._conn()
            c.executemany(sql, rows)
            c.commit()
            if time.time() - self._last_prune > 600:
                c.execute("DELETE FROM events WHERE ts < ?", (time.time() - self.retention_s,))
                c.commit()
                self._last_prune = time.time()

    # ------------------------------------------------------------- read
    def max_id(self) -> int:
        return self._rows("SELECT COALESCE(MAX(id), 0) AS m FROM events")[0]["m"]

    def last_event_ts(self) -> float:
        """Newest event time (POSIX seconds), where the scorer resumes after a restart; 0 when empty."""
        return float(self._rows("SELECT COALESCE(MAX(ts), 0) AS m FROM events")[0]["m"] or 0)

    def recent(self, limit: int = 50, since_id: int = 0) -> List[dict]:
        rows = self._rows("SELECT * FROM events WHERE id > ? ORDER BY id DESC LIMIT ?", (since_id, limit))
        for r in rows:
            r["reason_codes"] = [x for x in (r.get("reason_codes") or "").split(",") if x]
        return rows

    def totals(self) -> Dict:
        r = self._rows("SELECT COUNT(*) n, SUM(decision='ALLOW') a, SUM(decision='REVIEW') rv, SUM(decision='BLOCK') b "
                       "FROM events")[0]
        return {"events": r["n"] or 0, "ALLOW": r["a"] or 0, "REVIEW": r["rv"] or 0, "BLOCK": r["b"] or 0}

    def summary(self, minutes: int = 60, top: int = 10) -> Dict:
        since = time.time() - minutes * 60
        w = "WHERE ts >= ?"
        a = (since,)
        tot = self._rows(f"""SELECT COUNT(*) n, SUM(decision='ALLOW') allow, SUM(decision='REVIEW') review,
            SUM(decision='BLOCK') block, SUM(fraud_label) fraud, SUM(amount) amount,
            SUM(CASE WHEN decision='BLOCK' THEN amount ELSE 0 END) blocked_amount,
            SUM(CASE WHEN decision='BLOCK' AND fraud_label=1 THEN amount ELSE 0 END) fraud_amount_stopped,
            SUM(CASE WHEN decision='ALLOW' AND fraud_label=1 THEN amount ELSE 0 END) fraud_amount_missed,
            AVG(scored_ts - ts) * 1000 e2e_ms FROM events {w}""", a)[0]
        tot = {k: (v or 0) for k, v in tot.items()}
        bucket = 60 if minutes <= 180 else 300 if minutes <= 1440 else 3600
        ts = self._rows(f"""SELECT CAST(ts / {bucket} AS INTEGER) * {bucket} t, decision, COUNT(*) n
            FROM events {w} GROUP BY t, decision ORDER BY t""", a)

        def group(col: str, extra: str = "", n: int = top) -> List[dict]:
            return self._rows(f"""SELECT {col} key, COUNT(*) n, SUM(decision='REVIEW') review, SUM(decision='BLOCK') block,
                SUM(fraud_label) fraud, ROUND(SUM(amount), 2) amount, ROUND(AVG(probability), 4) avg_p
                FROM events {w} {extra} GROUP BY {col}
                ORDER BY (SUM(decision='BLOCK') * 2 + SUM(decision='REVIEW')) DESC, n DESC LIMIT {int(n)}""", a)

        risky = "AND decision != 'ALLOW'"
        reasons: Counter = Counter()
        for r in self._rows(f"SELECT reason_codes FROM events {w} AND decision != 'ALLOW'", a):
            reasons.update(x for x in (r["reason_codes"] or "").split(",") if x)
        models = {}
        for m in MODELS:
            r = self._rows(f"""SELECT COUNT({m}_p) n, AVG({m}_ms) avg_ms,
                SUM({m}_decision = decision) agree,
                SUM({m}_decision != 'ALLOW' AND fraud_label = 1) tp, SUM({m}_decision != 'ALLOW' AND fraud_label = 0) fp,
                SUM({m}_decision = 'ALLOW' AND fraud_label = 1) fn, SUM({m}_decision = 'BLOCK' AND fraud_label = 1) btp,
                SUM({m}_decision = 'BLOCK') bn FROM events {w} AND {m}_p IS NOT NULL""", a)[0]
            n = r["n"] or 0
            lat = [x[f"{m}_ms"] for x in self._rows(f"SELECT {m}_ms FROM events {w} AND {m}_ms IS NOT NULL "
                                                     f"ORDER BY {m}_ms", a)]
            tp, fp, fn = r["tp"] or 0, r["fp"] or 0, r["fn"] or 0
            models[m] = {"scored": n, "avg_ms": r["avg_ms"],
                         "p95_ms": lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else None,
                         "agreement": (r["agree"] or 0) / n if n else None,
                         "recall": tp / (tp + fn) if tp + fn else None,
                         "precision": tp / (tp + fp) if tp + fp else None,
                         "precision_block": (r["btp"] or 0) / r["bn"] if r["bn"] else None}
        return {
            "minutes": minutes, "bucket_s": bucket, "totals": tot, "timeseries": ts,
            "by_customer": group("customer_id", risky), "by_merchant": group("merchant_name", risky),
            "by_device": group("device_id", risky), "by_ip": group("ip", risky + " AND ip IS NOT NULL"),
            "by_ring": group("ring_id", "AND ring_id IS NOT NULL"),
            "by_mcc": group("mcc", n=20), "by_channel": group("channel", n=10), "by_city": group("city", n=20),
            "by_fraud_type": group("fraud_type", n=10), "by_source": group("source", n=5),
            "by_hour": self._rows(f"SELECT hour key, decision, COUNT(*) n FROM events {w} GROUP BY hour, decision", a),
            "reasons": dict(reasons.most_common()), "models": models,
        }

    def export_json(self, limit: int = 1000) -> str:
        return json.dumps(self.recent(limit))
