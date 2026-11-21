"""Real-time scorer: Kafka sentinel.transactions -> models -> decision -> action.

Champion / challenger:
  champion  (CHAMPION, default xgboost) decides. If it is down, the next healthy model decides;
            if every model is down the transaction goes to REVIEW (fail closed, never auto-approve).
  shadow    the other models score the same batch in parallel so the dashboard can compare them live.
Output:
  sentinel.decisions   every scored transaction (all model outputs)
  sentinel.actions     APPROVE / STEP_UP_AUTH / BLOCK_AND_ALERT (the "further action" hook)
  SNS (optional)       BLOCK actions are published to SNS_ACTIONS_TOPIC_ARN for downstream automation
  SQLite               analytics store read by the API
  CloudWatch           custom metrics every 60 s when CW_NAMESPACE is set
Delivery is at-least-once: offsets are committed after the batch is stored; the store ignores duplicates.
Restarts (Fargate has no persistent disk): when the store is empty the processor first replays
sentinel.decisions from the beginning of its retention, so the analytics history survives redeploys.
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

from sentinel_stream import TOPIC_ACTIONS, TOPIC_DECISIONS, TOPIC_TX, action_for, kafka_config
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
                data.append({"MetricName": "ConsumerLag", "Dimensions": dims, "Value": float(lag), "Unit": "Count"})
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
        logger.info("metrics %s", json.dumps({**self.c, "lag": lag}))
        self.reset()


class Processor(threading.Thread):
    def __init__(self, store: Store, bootstrap: str = "", clients: Optional[Dict[str, ModelClient]] = None,
                 champion: Optional[str] = None, group: str = "sentinel-scorer"):
        super().__init__(daemon=True, name="processor")
        self.store, self.bootstrap, self.group = store, bootstrap, group
        self.clients = clients or clients_from_env()
        self.champion = (champion or os.getenv("CHAMPION", "xgboost")).lower()
        self.batch = int(os.getenv("BATCH_SIZE", "32"))
        self.stop_evt = threading.Event()
        self.pool = ThreadPoolExecutor(max_workers=3)
        self.metrics = Metrics(os.getenv("CW_NAMESPACE", ""))
        self.sns_topic = os.getenv("SNS_ACTIONS_TOPIC_ARN", "")
        self.sns = None
        self.last_sns = 0.0
        self.state = {"kafka": "connecting", "consumed": 0, "produced_total": None, "lag": None,
                      "last_batch_ts": None, "started": time.time(), "champion": self.champion, "errors": 0}

    def _lag(self, consumer) -> None:
        try:
            parts = consumer.assignment()
            if not parts:
                return
            pos = {(p.topic, p.partition): p.offset for p in consumer.position(parts)}
            lag = total = 0
            for p in parts:
                lo, hi = consumer.get_watermark_offsets(p, timeout=2, cached=False)
                total += hi
                cur = pos.get((p.topic, p.partition), -1)
                lag += max(0, hi - (cur if cur >= 0 else hi))
            self.state["lag"], self.state["produced_total"] = lag, total
        except Exception as e:  # noqa: BLE001
            logger.debug("lag check failed: %s", e)

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
                producer.produce(TOPIC_DECISIONS, key=key, value=json.dumps(r, default=str).encode())
                producer.produce(TOPIC_ACTIONS, key=key, value=json.dumps(
                    {"event_id": r["event_id"], "customer_id": r.get("customer_id"), "decision": r["decision"],
                     "action": r["action"], "text": r["action_text"], "amount": r.get("amount")}).encode())
            producer.poll(0)
        self._notify(records)
        self.metrics.add(records)
        self.state["consumed"] += len(records)
        self.state["last_batch_ts"] = time.time()
        return records

    def rebuild(self, max_seconds: float = 90) -> int:
        """Empty store -> reload past decisions from Kafka (no consumer group, nothing committed)."""
        from confluent_kafka import OFFSET_BEGINNING, Consumer, TopicPartition

        if self.store.max_id() or os.getenv("REBUILD_FROM_KAFKA", "1") != "1":
            return 0
        c = Consumer(kafka_config(**{"group.id": f"{self.group}-rebuild", "enable.auto.commit": False}))
        try:
            meta = c.list_topics(TOPIC_DECISIONS, timeout=15).topics.get(TOPIC_DECISIONS)
            if not meta or meta.error or not meta.partitions:
                return 0
            parts = [TopicPartition(TOPIC_DECISIONS, p, OFFSET_BEGINNING) for p in meta.partitions]
            ends = {tp.partition: c.get_watermark_offsets(tp, timeout=10)[1] for tp in parts}
            todo = {p for p, hi in ends.items() if hi > 0}
            c.assign(parts)
            n, t0, batch = 0, time.time(), []
            while todo and time.time() - t0 < max_seconds:
                for m in c.consume(500, timeout=1.0):
                    if m.error():
                        continue
                    try:
                        batch.append(json.loads(m.value()))
                    except ValueError:
                        pass
                    if m.offset() + 1 >= ends[m.partition()]:
                        todo.discard(m.partition())
                if batch:
                    self.store.insert(batch)
                    n += len(batch)
                    batch = []
            logger.info("Rebuilt the analytics store from %s: %d decisions", TOPIC_DECISIONS, n)
            self.state["rebuilt"] = n
            return n
        finally:
            c.close()

    def run(self):
        from confluent_kafka import Consumer, KafkaException, Producer

        while not self.stop_evt.is_set():
            try:
                self.rebuild()
                consumer = Consumer(kafka_config(**{"group.id": self.group, "auto.offset.reset": "latest",
                                                    "enable.auto.commit": False, "session.timeout.ms": 45000}))
                producer = Producer(kafka_config(**{"linger.ms": 50, "acks": "all"}))
                consumer.subscribe([TOPIC_TX])
                logger.info("Consuming %s (champion=%s)", TOPIC_TX, self.champion)
                last_lag = 0.0
                while not self.stop_evt.is_set():
                    msgs = consumer.consume(num_messages=self.batch, timeout=1.0)
                    self.state["kafka"] = "connected"
                    events = []
                    for m in msgs:
                        if m.error():
                            logger.warning("kafka: %s", m.error())
                            continue
                        try:
                            events.append(json.loads(m.value()))
                        except ValueError:
                            logger.warning("skipping non-JSON message at offset %s", m.offset())
                    if events:
                        self.handle(events, producer)
                        consumer.commit(asynchronous=True)
                    if time.time() - last_lag > 5:
                        self._lag(consumer)
                        last_lag = time.time()
                    self.metrics.flush(self.state["lag"])
                consumer.close()
                producer.flush(5)
            except (KafkaException, RuntimeError) as e:
                self.state["kafka"] = f"error: {e}"
                self.state["errors"] += 1
                logger.error("processor error, retrying in 5 s: %s", e)
                time.sleep(5)

    def stop(self):
        self.stop_evt.set()
