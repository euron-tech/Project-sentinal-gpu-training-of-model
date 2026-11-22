"""Amazon Kinesis Data Streams: the event bus between the producer, the scorer and downstream consumers.

    KinesisProducer   buffered PutRecords with retry of throttled records. Same produce / poll / flush interface the
                      producer and scorer use, so they do not care which bus is underneath.
    ShardReader       polls every shard of a stream with GetRecords, follows resharding, and reports lag
                      (MillisBehindLatest).
Env: AWS_REGION, KINESIS_ENDPOINT (local LocalStack / moto only), KINESIS_CREATE=1 (create missing streams: local only;
on AWS Terraform owns the streams).
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("sentinel.bus")

MAX_BATCH = 500                    # PutRecords limit per call
FLUSH_EVERY_S = 0.2


def kinesis_client():
    import boto3
    from botocore.config import Config

    return boto3.client("kinesis", endpoint_url=os.getenv("KINESIS_ENDPOINT") or None,
                        config=Config(retries={"max_attempts": 8, "mode": "adaptive"}, connect_timeout=5, read_timeout=20))


def ensure_streams(client, names: List[str], shards: int = 1, wait_s: float = 60) -> None:
    """Create any missing stream and wait until it is ACTIVE (local development; Terraform does this on AWS)."""
    for name in names:
        try:
            client.describe_stream_summary(StreamName=name)
        except client.exceptions.ResourceNotFoundException:
            client.create_stream(StreamName=name, ShardCount=shards)
            logger.info("created Kinesis stream %s", name)
    deadline = time.time() + wait_s
    for name in names:
        while time.time() < deadline:
            if client.describe_stream_summary(StreamName=name)["StreamDescriptionSummary"]["StreamStatus"] == "ACTIVE":
                break
            time.sleep(0.5)


def _key(key) -> str:
    k = key.decode() if isinstance(key, bytes) else str(key or "")
    return (k or "none")[:256]


class KinesisProducer:
    """produce() buffers; poll() sends when the buffer is big or old; flush() sends everything now."""

    def __init__(self, client=None):
        self.client = client or kinesis_client()
        self.buf: Dict[str, List[dict]] = {}
        self.last = time.time()
        self.sent = self.failed = 0
        if os.getenv("KINESIS_CREATE") == "1":
            from sentinel_stream import STREAM_DECISIONS, STREAM_TX
            ensure_streams(self.client, [STREAM_TX, STREAM_DECISIONS])

    def produce(self, stream: str, key=None, value=None) -> None:
        data = value if isinstance(value, bytes) else json.dumps(value, default=str).encode()
        self.buf.setdefault(stream, []).append({"Data": data, "PartitionKey": _key(key)})

    def poll(self, _timeout: float = 0) -> int:
        n = sum(len(v) for v in self.buf.values())
        if n >= MAX_BATCH or (n and time.time() - self.last >= FLUSH_EVERY_S):
            return self.flush()
        return 0

    def flush(self, _timeout: Optional[float] = None) -> int:
        n = 0
        for stream, recs in list(self.buf.items()):
            for i in range(0, len(recs), MAX_BATCH):
                n += self._put(stream, recs[i:i + MAX_BATCH])
        self.buf.clear()
        self.last = time.time()
        return n

    def _put(self, stream: str, recs: List[dict]) -> int:
        for attempt in range(5):
            r = self.client.put_records(StreamName=stream, Records=recs)
            if not r.get("FailedRecordCount"):
                self.sent += len(recs)
                return len(recs)
            # keep only the throttled / failed records and retry them with back-off
            failed = [rec for rec, res in zip(recs, r["Records"]) if res.get("ErrorCode")]
            self.sent += len(recs) - len(failed)
            recs = failed
            time.sleep(0.2 * 2 ** attempt)
        self.failed += len(recs)
        logger.warning("dropped %d records for %s after retries", len(recs), stream)
        return 0


class ShardReader:
    """Reads every shard of one stream. start: LATEST | TRIM_HORIZON | a POSIX timestamp (AT_TIMESTAMP)."""

    def __init__(self, stream: str, client=None, start="LATEST", limit: int = 100):
        self.stream, self.client, self.limit = stream, client or kinesis_client(), limit
        self.start = start
        self.its: Dict[str, Optional[str]] = {}       # shard id -> iterator (None = shard closed and drained)
        self.seq: Dict[str, str] = {}                 # shard id -> last sequence number read
        self.lag_ms: Dict[str, int] = {}
        self.listed = 0.0
        self.next_call: Dict[str, float] = {}

    def _iterator(self, shard: str) -> str:
        kw: dict = {"StreamName": self.stream, "ShardId": shard}
        if shard in self.seq:
            kw.update(ShardIteratorType="AFTER_SEQUENCE_NUMBER", StartingSequenceNumber=self.seq[shard])
        elif isinstance(self.start, (int, float)):
            kw.update(ShardIteratorType="AT_TIMESTAMP", Timestamp=datetime.fromtimestamp(self.start, timezone.utc))
        else:
            kw.update(ShardIteratorType=self.start)
        return self.client.get_shard_iterator(**kw)["ShardIterator"]

    def _list(self) -> None:
        """Pick up new shards (resharding). Re-listed every 60 s."""
        shards, token = [], None
        while True:
            r = self.client.list_shards(NextToken=token) if token else self.client.list_shards(StreamName=self.stream)
            shards += r["Shards"]
            token = r.get("NextToken")
            if not token:
                break
        for s in shards:
            if s["ShardId"] not in self.its:
                self.its[s["ShardId"]] = self._iterator(s["ShardId"])
        self.listed = time.time()

    @property
    def lag(self) -> Optional[int]:
        return max(self.lag_ms.values()) if self.lag_ms else None

    def read(self) -> Tuple[List[dict], bool]:
        """One GetRecords per open shard (max 5 calls / s / shard). Returns (decoded records, caught_up)."""
        if not self.its or time.time() - self.listed > 60:
            self._list()
        out, caught_up = [], True
        for shard, it in list(self.its.items()):
            if it is None or time.time() < self.next_call.get(shard, 0):
                continue
            try:
                r = self.client.get_records(ShardIterator=it, Limit=self.limit)
            except self.client.exceptions.ExpiredIteratorException:
                self.its[shard] = self._iterator(shard)
                continue
            except self.client.exceptions.ProvisionedThroughputExceededException:
                self.next_call[shard] = time.time() + 1
                caught_up = False
                continue
            self.next_call[shard] = time.time() + 0.21
            for rec in r["Records"]:
                self.seq[shard] = rec["SequenceNumber"]
                try:
                    out.append(json.loads(rec["Data"]))
                except ValueError:
                    logger.warning("skipping non-JSON record %s", rec["SequenceNumber"])
            self.its[shard] = r.get("NextShardIterator")       # None once a closed (resharded) shard is drained
            self.lag_ms[shard] = int(r.get("MillisBehindLatest") or 0)
            if r["Records"] or self.lag_ms[shard] > 0:
                caught_up = False
        return out, caught_up
