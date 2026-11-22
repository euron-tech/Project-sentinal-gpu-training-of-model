"""Auto-mode traffic generator: publishes transactions to Amazon Kinesis (stream sentinel-<env>-transactions) while switched on.

On AWS it runs inside the stream service and is switched on / off with the "Start auto traffic" button on the
dashboard's Live Flow tab (POST /live/api/producer). It is OFF until someone presses the button
(AUTO_START=1 starts it at boot).

Standalone (keeps sending until Ctrl-C):
    python -m sentinel_stream.producer
Env:
  STREAM_TRANSACTIONS, AWS_REGION   (KINESIS_ENDPOINT + KINESIS_CREATE=1 for a local LocalStack)
  EVENTS_PER_SEC    2           steady rate (Poisson arrivals)
  FRAUD_RATE        0.04        share of synthetic fraud
  ATTACK_EVERY_S    300         while on, a fraud ring fires an attack wave every N seconds (0 = off)
  ATTACK_SIZE       15
  PRODUCER_SOURCE   mix         synthetic | dataset | mix   (dataset/mix need DATASET_URI)
  DATASET_URI                   local dir or s3://bucket/prefix with *.jsonl.gz (labelled SFT holdout split)
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from typing import Optional

from sentinel_stream import STREAM_TX
from sentinel_stream.generator import make_source

logger = logging.getLogger("sentinel.producer")


def make_producer():
    from sentinel_stream.bus import KinesisProducer

    return KinesisProducer()


def send(producer, events) -> int:
    for e in events:
        producer.produce(STREAM_TX, key=(e.get("customer_id") or "").encode(), value=json.dumps(e).encode())
    producer.poll(0)
    return len(events)


class Traffic:
    """Start / stop / re-rate a background thread that keeps sending transactions."""

    def __init__(self, producer=None, source=None):
        self._producer, self._source = producer, source
        self.rate = float(os.getenv("EVENTS_PER_SEC", "2"))
        self.attack_every = float(os.getenv("ATTACK_EVERY_S", "300"))
        self.attack_size = int(os.getenv("ATTACK_SIZE", "15"))
        self.sent = self.attacks = 0
        self.started_at: Optional[float] = None
        self.error: Optional[str] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    @property
    def producer(self):
        if self._producer is None:
            self._producer = make_producer()
        return self._producer

    @property
    def source(self):
        if self._source is None:
            self._source = make_source(os.getenv("PRODUCER_SOURCE", "mix"))
        return self._source

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def status(self) -> dict:
        return {"running": self.running, "rate": self.rate, "sent": self.sent, "attacks": self.attacks,
                "source": type(self._source).__name__ if self._source else os.getenv("PRODUCER_SOURCE", "mix"),
                "started_at": self.started_at, "error": self.error}

    def start(self, rate: Optional[float] = None) -> dict:
        with self._lock:
            if rate is not None:
                self.rate = max(0.1, min(50.0, float(rate)))
            if not self.running:
                self._stop.clear()
                self.error = None
                self.started_at = time.time()
                self._thread = threading.Thread(target=self._loop, daemon=True, name="traffic")
                self._thread.start()
                logger.info("Auto traffic ON at %.1f tx/s", self.rate)
        return self.status()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Auto traffic OFF after %d transactions", self.sent)
        return self.status()

    def burst(self, n: int, fraud_type: Optional[str] = None) -> int:
        k = send(self.producer, self.source.burst(n, fraud_type))
        self.producer.flush(5)
        self.sent += k
        self.attacks += 1
        return k

    def _loop(self):
        try:
            next_attack = time.time() + self.attack_every
            while not self._stop.is_set():
                self.sent += send(self.producer, [self.source.next()])
                if self.attack_every and time.time() >= next_attack:
                    self.burst(self.attack_size)
                    next_attack = time.time() + self.attack_every * random.uniform(0.7, 1.3)
                self._stop.wait(random.expovariate(self.rate))
            self.producer.flush(5)
        except Exception as e:  # noqa: BLE001 - surface the error in the UI instead of dying silently
            self.error = f"{type(e).__name__}: {e}"
            logger.exception("traffic generator stopped")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    t = Traffic()
    t.start()
    try:
        while t.running:
            time.sleep(60)
            logger.info("sent %d transactions", t.sent)
    except KeyboardInterrupt:
        t.stop()


if __name__ == "__main__":
    main()
