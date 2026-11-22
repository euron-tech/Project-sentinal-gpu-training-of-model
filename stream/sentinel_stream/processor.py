"""Real-time scorer: Kinesis stream <env>-transactions -> models -> decision -> action.

Champion / challenger:
  champion  (CHAMPION, default xgboost) decides. If it is down, the next healthy model decides;
            if every model is down the transaction goes to REVIEW (fail closed, never auto-approve).
  shadow    the other models score the same batch in parallel so the dashboard can compare them live.
Output:
  Kinesis <env>-decisions   every scored transaction: all model outputs + the action (APPROVE / STEP_UP_AUTH /
                            BLOCK_AND_ALERT), the "further action" hook for downstream consumers; kept 24 h
  SNS (optional)            BLOCK actions are published to SNS_ACTIONS_TOPIC_ARN for downstream automation
  SQLite                    analytics store read by the API
  CloudWatch                custom metrics every 60 s when CW_NAMESPACE is set
Delivery is at-least-once: a batch is stored before it is acknowledged; the store ignores duplicates (event_id).
Restarts (Fargate has no persistent disk): when the store is empty the processor first replays the decisions
stream from the start of its retention, then resumes the transactions stream just before the newest decision,
so the analytics history survives redeploys and in-flight transactions are not lost.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import requests

from sentinel_stream import STREAM_DECISIONS, STREAM_TX, action_for
from sentinel_stream.store import MODELS, Store

logger = logging.getLogger("sentinel.processor")


class ModelClient:
    """POST /score client with health tracking and back-off, so a dead model never stalls the stream."""

    def __init__(self, name: str, url: str, key: str = "", timeout: float = 10.0):
        self.name, self.url, self.key, self.timeout = name, (url or "").rstrip("/"), key, timeout
        self.session = requests.Session()
        self.healthy: Optional[bool] = None
        self.status = "unknown"
        self.retry_at = 0.0
        self.scored = self.errors = 0
        self.last_ms: Optional[float] = None

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    def check(self) -> bool:
        if not self.enabled:
            self.healthy, self.status = False, "disabled"
            return False
        try:
            r = self.session.get(self.url + "/health", timeout=3)
            self.status = r.json().get("status", "?") if r.ok else f"HTTP {r.status_code}"
            self.healthy = r.ok and self.status == "ok"
        except (requests.RequestException, ValueError) as e:
            self.healthy, self.status = False, type(e).__name__
        self.retry_at = 0.0 if self.healthy else time.time() + 30
        return bool(self.healthy)

    def ready(self) -> bool:
        if self.healthy:
            return True
        return time.time() >= self.retry_at and self.check()

    def score(self, items: List[dict]) -> Tuple[Optional[List[dict]], float]:
        t0 = time.perf_counter()
        try:
            r = self.session.post(self.url + "/score", json={"transactions": items}, timeout=self.timeout,
                                  headers={"X-API-Key": self.key} if self.key else {})
            r.raise_for_status()
            res = r.json()["results"]
            ms = (time.perf_counter() - t0) * 1000
            self.scored += len(items)
            self.last_ms = ms
            return res, ms
        except (requests.RequestException, KeyError, ValueError) as e:
            self.errors += 1
            self.healthy, self.status, self.retry_at = False, f"error: {type(e).__name__}", time.time() + 15
            logger.warning("%s scoring failed: %s", self.name, e)
            return None, (time.perf_counter() - t0) * 1000


def clients_from_env() -> Dict[str, ModelClient]:
    key = os.getenv("SENTINEL_API_KEY", "")
    return {
        "xgboost": ModelClient("xgboost", os.getenv("XGB_URL", "http://localhost:8001"), os.getenv("XGB_API_KEY", key)),
        "lora": ModelClient("lora", os.getenv("LORA_URL", ""), os.getenv("LLM_API_KEY", key), timeout=60),
        "qlora": ModelClient("qlora", os.getenv("QLORA_URL", ""), os.getenv("LLM_API_KEY", key), timeout=60),
    }


def score_batch(events: List[dict], clients: Dict[str, ModelClient], champion: str = "xgboost",
                pool: Optional[ThreadPoolExecutor] = None) -> List[dict]:
    """Score events on every ready model; the champion (or first healthy fallback) decides. Pure apart from HTTP."""
    items = [e["transaction"] for e in events]
    ready = [n for n in MODELS if n in clients and clients[n].ready()]
    own = pool is None
    pool = pool or ThreadPoolExecutor(max_workers=3)
    try:
        futures = {n: pool.submit(clients[n].score, items) for n in ready}
        outputs = {n: f.result() for n, f in futures.items()}
    finally:
        if own:
            pool.shutdown(wait=False)
    order = [champion] + [m for m in MODELS if m != champion]
    decider = next((m for m in order if outputs.get(m, (None, 0))[0] is not None), None)
    now = time.time()
    records = []
    for i, e in enumerate(events):
        rec = {k: e.get(k) for k in ("event_id", "ts", "source", "customer_id", "merchant_id", "merchant_name", "device_id",
                                     "ip", "ring_id", "amount", "mcc", "channel", "city", "merchant_city", "hour",
                                     "fraud_label", "fraud_type")}
        for m in MODELS:
            res, ms = outputs.get(m, (None, None))
            if res is not None:
                rec[f"{m}_p"], rec[f"{m}_decision"] = res[i]["fraud_probability"], res[i]["decision"]
                rec[f"{m}_ms"] = round(ms / len(events), 2)
        if decider:
            out = outputs[decider][0][i]
            rec.update(champion=decider, decision=out["decision"], probability=out["fraud_probability"],
                       reason_codes=out["reason_codes"], explanation=out["explanation"])
        else:
            rec.update(champion="none", decision="REVIEW", probability=None, reason_codes=[],
                       explanation="No scoring model available - held for manual review (fail closed).")
        act = action_for(rec["decision"])
        rec["action"] = act["action"]
        rec["action_text"] = act["text"]
        rec["scored_ts"] = now
        records.append(rec)
    return records


class Metrics:
    """Aggregates counters and pushes them to CloudWatch (namespace CW_NAMESPACE) once a minute."""

    def __init__(self, namespace: str = "", service: str = "stream"):
        self.namespace, self.service = namespace, service
        self.reset()
        self.cw = None
        if namespace:
            try:
                import boto3
                self.cw = boto3.client("cloudwatch")
            except Exception as e:  # noqa: BLE001
                logger.warning("CloudWatch metrics disabled: %s", e)

    def reset(self):
        self.c = {"Transactions": 0, "Allowed": 0, "Reviewed": 0, "Blocked": 0, "LabelledFraud": 0, "FraudMissed": 0, "NoModel": 0}
        self.lat: Dict[str, List[float]] = {m: [] for m in MODELS}
        self.t0 = time.time()

    def add(self, records: List[dict]):
        for r in records:
            self.c["Transactions"] += 1
            self.c[{"ALLOW": "Allowed", "REVIEW": "Reviewed", "BLOCK": "Blocked"}[r["decision"]]] += 1
            self.c["NoModel"] += int(r.get("champion") == "none")
            if r.get("fraud_label"):
                self.c["LabelledFraud"] += 1
                self.c["FraudMissed"] += int(r["decision"] == "ALLOW")
            for m in MODELS:
                if r.get(f"{m}_ms") is not None:
                    self.lat[m].append(r[f"{m}_ms"])

    def flush(self, lag: Optional[int] = None, force: bool = False):
        if not force and time.time() - self.t0 < 60:
            return
        if self.cw:
            dims = [{"Name": "Service", "Value": self.service}]
            data = [{"MetricName": k, "Dimensions": dims, "Value": float(v), "Unit": "Count"} for k, v in self.c.items()]
            if lag is not None:
                data.append({"MetricName": "StreamLagMs", "Dimensions": dims, "Value": float(lag), "Unit": "Milliseconds"})
            for m, v in self.lat.items():
                if v:
                    data.append({"MetricName": "ScoreLatency", "Unit": "Milliseconds",
                                 "Dimensions": [{"Name": "Model", "Value": m}],
                                 "StatisticValues": {"SampleCount": len(v), "Sum": sum(v), "Minimum": min(v), "Maximum": max(v)}})
            try:
                for i in range(0, len(data), 20):
                    self.cw.put_metric_data(Namespace=self.namespace, MetricData=data[i:i + 20])
            except Exception as e:  # noqa: BLE001 - metrics must never stop scoring
                logger.warning("put_metric_data failed: %s", e)
        logger.info("metrics %s", json.dumps({**self.c, "lag_ms": lag}))
        self.reset()


class Processor(threading.Thread):
    def __init__(self, store: Store, clients: Optional[Dict[str, ModelClient]] = None,
                 champion: Optional[str] = None, kinesis=None):
        super().__init__(daemon=True, name="processor")
        self.store, self.kinesis = store, kinesis
        self.clients = clients or clients_from_env()
        self.champion = (champion or os.getenv("CHAMPION", "xgboost")).lower()
        self.batch = int(os.getenv("BATCH_SIZE", "32"))
        self.stop_evt = threading.Event()
        self.pool = ThreadPoolExecutor(max_workers=3)
        self.metrics = Metrics(os.getenv("CW_NAMESPACE", ""))
        self.sns_topic = os.getenv("SNS_ACTIONS_TOPIC_ARN", "")
        self.sns = None
        self.last_sns = 0.0
        self.state = {"stream": "connecting", "consumed": 0, "lag_ms": None,
                      "last_batch_ts": None, "started": time.time(), "champion": self.champion, "errors": 0}

    def _notify(self, records: List[dict]):
        blocks = [r for r in records if r["decision"] == "BLOCK"]
        if not blocks or not self.sns_topic or time.time() - self.last_sns < 1:
            return
        try:
            if self.sns is None:
                import boto3
                self.sns = boto3.client("sns")
            msg = {"action": "BLOCK_AND_ALERT", "count": len(blocks),
                   "transactions": [{k: r.get(k) for k in ("event_id", "customer_id", "merchant_name", "amount",
                                                           "probability", "reason_codes")} for r in blocks[:10]]}
            self.sns.publish(TopicArn=self.sns_topic, Subject="SENTINEL: fraud blocked", Message=json.dumps(msg, default=str))
            self.last_sns = time.time()
        except Exception as e:  # noqa: BLE001
            logger.warning("SNS publish failed: %s", e)

    def handle(self, events: List[dict], producer=None) -> List[dict]:
        records = score_batch(events, self.clients, self.champion, self.pool)
        self.store.insert(records)
        if producer is not None:
            for r in records:
                key = (r.get("customer_id") or "").encode()
                producer.produce(STREAM_DECISIONS, key=key, value=json.dumps(r, default=str).encode())
            producer.poll(0)
        self._notify(records)
        self.metrics.add(records)
        self.state["consumed"] += len(records)
        self.state["last_batch_ts"] = time.time()
        return records

    def _client(self):
        if self.kinesis is None:
            from sentinel_stream.bus import kinesis_client
            self.kinesis = kinesis_client()
        return self.kinesis

    def rebuild(self, max_seconds: float = 120) -> int:
        """Empty store -> reload past decisions from the decisions stream (TRIM_HORIZON, read-only)."""
        from sentinel_stream.bus import ShardReader

        if self.store.max_id() or os.getenv("REBUILD_FROM_STREAM", "1") != "1":
            return 0
        reader = ShardReader(STREAM_DECISIONS, self._client(), start="TRIM_HORIZON", limit=1000)
        n, t0 = 0, time.time()
        while time.time() - t0 < max_seconds:
            recs, caught_up = reader.read()
            if recs:
                self.store.insert(recs)
                n += len(recs)
            if caught_up:
                break
            time.sleep(0.05 if recs else 0.25)
        logger.info("Rebuilt the analytics store from %s: %d decisions", STREAM_DECISIONS, n)
        self.state["rebuilt"] = n
        return n

    def run(self):
        from botocore.exceptions import BotoCoreError, ClientError

        from sentinel_stream.bus import KinesisProducer, ShardReader

        while not self.stop_evt.is_set():
            try:
                producer = KinesisProducer(self._client())   # KINESIS_CREATE=1 creates the streams locally
                self.rebuild()
                last = self.store.last_event_ts()
                # resume just before the newest decision (duplicates are ignored), else only new traffic
                reader = ShardReader(STREAM_TX, self._client(), start=(last - 5) if last else "LATEST", limit=self.batch)
                logger.info("Consuming Kinesis %s (champion=%s)", STREAM_TX, self.champion)
                while not self.stop_evt.is_set():
                    events, _ = reader.read()
                    self.state["stream"] = "connected"
                    for i in range(0, len(events), self.batch):
                        self.handle(events[i:i + self.batch], producer)
                    producer.poll(0)
                    self.state["lag_ms"] = reader.lag
                    self.metrics.flush(reader.lag)
                    if not events:
                        self.stop_evt.wait(0.25)
                producer.flush()
            except (BotoCoreError, ClientError, RuntimeError) as e:
                self.state["stream"] = f"error: {type(e).__name__}"
                self.state["errors"] += 1
                logger.error("processor error, retrying in 5 s: %s", e)
                time.sleep(5)

    def stop(self):
        self.stop_evt.set()
