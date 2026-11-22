"""Transaction sources for the Kinesis producer.

SyntheticSource  realistic customers, merchants and devices, with fraud rings that inject
                 account takeover, card testing, mule accounts and impossible travel.
                 Every event carries ground truth (fraud_label, fraud_type) so live accuracy can be measured.
DatasetSource    replays the labelled SFT dataset (fmt-v1 prompts) from a local dir or s3://bucket/prefix.

Both yield events of the same shape:
  {event_id, ts, source, customer_id, merchant_id, device_id, ip, fraud_label, fraud_type,
   amount, mcc, channel, city, merchant_city, hour, transaction: {...fields for /score...}}
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import math
import os
import random
import re
import time
import uuid
from pathlib import Path
from typing import Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

CITIES = ["Mumbai", "Delhi", "Bengaluru", "Pune", "Hyderabad", "Chennai", "Kolkata", "Ahmedabad",
          "Jaipur", "Lucknow", "Patna", "Kochi"]
MCC_RANGE = {"grocery": (150, 2500), "quick_commerce": (100, 1500), "food": (120, 1800), "electronics": (2000, 60000),
             "gaming": (100, 5000), "gift_cards": (500, 10000), "healthcare": (200, 8000), "travel": (1500, 40000),
             "education": (1000, 30000), "other": (100, 5000)}
GENUINE_MCC_W = {"grocery": 26, "quick_commerce": 16, "food": 18, "electronics": 5, "gaming": 4, "gift_cards": 2,
                 "healthcare": 7, "travel": 5, "education": 4, "other": 13}
CHANNELS = ["UPI", "card", "wallet", "net_banking"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
FRAUD_TYPES = ("account_takeover", "card_testing", "mule_account", "impossible_travel")
MERCHANT_WORDS = ["Bazaar", "Mart", "Store", "Hub", "Express", "Digital", "World", "Point", "Depot", "Corner"]


def _dist_bucket(a: str, b: str, rnd: random.Random) -> str:
    return "<5" if a == b else rnd.choice(["200-500", "500-1000", "1000+"])


class SyntheticSource:
    def __init__(self, fraud_rate: float = 0.04, seed: Optional[int] = None, n_customers: int = 2000,
                 n_merchants: int = 300, n_rings: int = 8):
        self.rnd = random.Random(seed)
        self.fraud_rate = fraud_rate
        r = self.rnd
        self.customers = []
        for i in range(n_customers):
            avg = round(math.exp(r.gauss(6.8, 0.7)), 2)            # median ~900 INR
            self.customers.append({
                "customer_id": f"CUST-{i:05d}", "age": r.randint(19, 72), "account_age_days": r.randint(30, 3650),
                "home_city": r.choice(CITIES), "avg": avg, "device_id": f"DEV-{i:05d}",
                "ip": f"49.{r.randint(32, 63)}.{r.randint(0, 255)}.{r.randint(1, 254)}",
                "channel": r.choices(CHANNELS, [60, 22, 12, 6])[0], "txns_30d": r.randint(3, 45)})
        self.merchants = []
        for i in range(n_merchants):
            mcc = r.choices(list(GENUINE_MCC_W), list(GENUINE_MCC_W.values()))[0]
            self.merchants.append({"merchant_id": f"MER-{i:04d}", "mcc": mcc, "city": r.choice(CITIES),
                                   "name": f"{r.choice(CITIES)} {mcc.replace('_', ' ').title()} {r.choice(MERCHANT_WORDS)}"})
        # Fraud rings: a handful of devices / IPs / mule merchants that commit most of the fraud.
        self.rings = []
        for i in range(n_rings):
            self.rings.append({"ring_id": f"RING-{i:02d}", "device_id": f"DEV-ATK-{i:02d}",
                               "ip": f"185.{r.randint(100, 220)}.{r.randint(0, 255)}.{r.randint(1, 254)}",
                               "city": r.choice(CITIES), "type": FRAUD_TYPES[i % len(FRAUD_TYPES)],
                               "merchant": {"merchant_id": f"MER-X{i:02d}", "mcc": r.choice(["electronics", "gift_cards", "gaming"]),
                                            "city": r.choice(CITIES), "name": f"Quick{r.choice(MERCHANT_WORDS)} Online {i}"}})
        self.by_mcc: Dict[str, List[dict]] = {}
        for m in self.merchants:
            self.by_mcc.setdefault(m["mcc"], []).append(m)

    # -------------------------------------------------------------- helpers
    def _history(self, c: dict, n: int) -> List[dict]:
        r = self.rnd
        out = []
        for _ in range(n):
            m = r.choice(self.merchants)
            lo, hi = MCC_RANGE[m["mcc"]]
            out.append({"days_ago": r.randint(1, 60), "channel": c["channel"],
                        "amount": round(min(hi, max(lo, r.gauss(c["avg"], c["avg"] * 0.4))), 1),
                        "mcc": m["mcc"], "city": c["home_city"]})
        return sorted(out, key=lambda h: h["days_ago"])

    def _event(self, c: dict, m: dict, amount: float, label: int, ftype: str, **over) -> dict:
        r = self.rnd
        hour = over.pop("hour", r.choices(range(24), [1, 1, 1, 1, 1, 2, 3, 5, 7, 8, 9, 9, 10, 9, 8, 8, 8, 9, 10, 10, 9, 7, 4, 2])[0])
        city = over.pop("city", c["home_city"])
        tx = {"transaction_id": "TX-" + uuid.uuid4().hex[:12].upper(), "amount": round(amount, 2), "mcc": m["mcc"],
              "channel": over.pop("channel", c["channel"]), "hour": hour, "weekday": DAYS[time.gmtime().tm_wday],
              "customer_age": c["age"], "account_age_days": over.pop("account_age_days", c["account_age_days"]),
              "avg_90d": over.pop("avg_90d", c["avg"]), "txns_30d": over.pop("txns_30d", c["txns_30d"]),
              "fails_24h": over.pop("fails_24h", 0), "new_device": over.pop("new_device", False),
              "city": city, "merchant_city": m["city"], "new_city": over.pop("new_city", False),
              "dist_km": over.pop("dist_km", _dist_bucket(city, m["city"], r)),
              "txns_1h": over.pop("txns_1h", r.choices([0, 1, 2], [70, 25, 5])[0]),
              "amount_24h": over.pop("amount_24h", round(r.uniform(0, 2) * c["avg"], 2)),
              "device_users_7d": over.pop("device_users_7d", 1),
              "history": over.pop("history", self._history(c, r.randint(2, 6)))}
        avg = tx["avg_90d"]
        tx["amt_ratio"] = round(amount / avg, 1) if avg else None
        return {"event_id": uuid.uuid4().hex, "ts": time.time(), "source": "synthetic",
                "customer_id": c["customer_id"], "merchant_id": m["merchant_id"], "merchant_name": m.get("name", m["merchant_id"]),
                "device_id": over.pop("device_id", c["device_id"]), "ip": over.pop("ip", c["ip"]),
                "fraud_label": label, "fraud_type": ftype, "ring_id": over.pop("ring_id", None),
                "amount": tx["amount"], "mcc": tx["mcc"], "channel": tx["channel"], "city": city,
                "merchant_city": m["city"], "hour": hour, "transaction": tx}

    # -------------------------------------------------------------- generators
    def genuine(self) -> dict:
        r = self.rnd
        c = r.choice(self.customers)
        mcc = r.choices(list(GENUINE_MCC_W), list(GENUINE_MCC_W.values()))[0]
        local = [m for m in self.by_mcc.get(mcc, self.merchants) if m["city"] == c["home_city"]]
        m = r.choice(local or self.by_mcc.get(mcc) or self.merchants)
        lo, hi = MCC_RANGE[mcc]
        amount = min(hi, max(lo, r.lognormvariate(math.log(max(lo, c["avg"])), 0.6)))
        travel = r.random() < 0.04                                    # benign anomalies -> realistic false positives
        over = {}
        if travel:
            over.update(city=r.choice(CITIES), new_city=True)
        if r.random() < 0.03:
            over.update(new_device=True)
        if r.random() < 0.02:
            over.update(fails_24h=1)
        return self._event(c, m, amount, 0, "genuine", **over)

    def fraud(self, ftype: Optional[str] = None, ring: Optional[dict] = None) -> dict:
        r = self.rnd
        ring = ring or r.choice(self.rings)
        ftype = ftype or ring["type"]
        c = r.choice(self.customers)
        m = ring["merchant"] if r.random() < 0.6 else r.choice(self.by_mcc.get("electronics", self.merchants))
        base = {"device_id": ring["device_id"], "ip": ring["ip"], "ring_id": ring["ring_id"]}
        if ftype == "account_takeover":
            amount = c["avg"] * r.uniform(8, 45)
            over = dict(new_device=True, new_city=True, city=ring["city"], hour=r.choice([0, 1, 2, 3, 4, 23]),
                        fails_24h=r.randint(2, 5), device_users_7d=r.randint(3, 8), channel="card")
        elif ftype == "card_testing":
            amount = r.uniform(1, 250)
            m = ring["merchant"]
            over = dict(txns_1h=r.randint(6, 15), fails_24h=r.randint(1, 4), channel="card", new_device=r.random() < 0.7,
                        device_users_7d=r.randint(4, 12), amount_24h=round(r.uniform(500, 3000), 2))
        elif ftype == "mule_account":
            amount = r.uniform(20000, 95000)
            over = dict(account_age_days=r.randint(1, 20), avg_90d=None, txns_30d=r.randint(0, 3), history=[],
                        channel=r.choice(["UPI", "net_banking"]), new_device=True)
        else:  # impossible_travel
            amount = c["avg"] * r.uniform(3, 15)
            far = r.choice([x for x in CITIES if x != c["home_city"]])
            hist = self._history(c, 3)
            hist.insert(0, {"days_ago": 0, "channel": c["channel"], "amount": round(c["avg"], 1), "mcc": "food", "city": c["home_city"]})
            over = dict(city=far, new_city=True, dist_km="1000+", history=hist, new_device=r.random() < 0.5)
        return self._event(c, m, amount, 1, ftype, **base, **over)

    def next(self) -> dict:
        return self.fraud() if self.rnd.random() < self.fraud_rate else self.genuine()

    def burst(self, n: int = 20, ftype: Optional[str] = None) -> List[dict]:
        """An attack wave: one ring fires n fraud transactions (used by the producer and /inject)."""
        ring = self.rnd.choice([g for g in self.rings if not ftype or g["type"] == ftype] or self.rings)
        return [self.fraud(ftype or ring["type"], ring) for _ in range(n)]

    def __iter__(self) -> Iterator[dict]:
        while True:
            yield self.next()


# ------------------------------------------------------------------ dataset replay
_RX = {k: re.compile(p) for k, p in {
    "amount": r"amt=([\d.]+)", "mcc": r"mcc=(\S+)", "channel": r"channel=(\S+)", "hour": r"time=(\d+)h",
    "city": r"\bcity=(\S+)", "merchant_city": r"merchant_city=(\S+)"}.items()}


def prompt_meta(prompt: str) -> dict:
    out = {}
    for k, rx in _RX.items():
        m = rx.search(prompt)
        out[k] = m.group(1) if m else None
    out["amount"] = float(out["amount"]) if out["amount"] else 0.0
    out["hour"] = int(out["hour"]) if out["hour"] else None
    return out


def _sync_s3(uri: str, dest: Path) -> Path:
    import boto3

    bucket, _, prefix = uri[5:].partition("/")
    s3 = boto3.client("s3")
    dest.mkdir(parents=True, exist_ok=True)
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for o in page.get("Contents", []):
            if o["Key"].endswith(".jsonl.gz"):
                f = dest / Path(o["Key"]).name
                if not f.exists():
                    s3.download_file(bucket, o["Key"], str(f))
    return dest


class DatasetSource:
    """Replays labelled rows; customer/merchant ids are derived from the prompt so analytics can group them."""

    def __init__(self, location: str, seed: Optional[int] = None, max_rows: int = 200_000):
        path = _sync_s3(location, Path("/tmp/sentinel-dataset")) if location.startswith("s3://") else Path(location)
        self.rows: List[dict] = []
        for fp in sorted(path.glob("*.jsonl.gz")):
            with gzip.open(fp, "rt", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        self.rows.append({"transaction_id": r.get("transaction_id") or r.get("example_id"),
                                          "prompt": r["prompt"], "fraud_label": int(r["fraud_label"])})
                        if len(self.rows) >= max_rows:
                            break
            if len(self.rows) >= max_rows:
                break
        if not self.rows:
            raise FileNotFoundError(f"no *.jsonl.gz rows under {location}")
        self.rnd = random.Random(seed)
        logger.info("Dataset replay: %d rows (%d fraud) from %s", len(self.rows),
                    sum(r["fraud_label"] for r in self.rows), location)

    def next(self) -> dict:
        row = self.rnd.choice(self.rows)
        meta = prompt_meta(row["prompt"])
        h = hashlib.sha1(row["transaction_id"].encode()).hexdigest()
        return {"event_id": uuid.uuid4().hex, "ts": time.time(), "source": "dataset",
                "customer_id": f"CUST-D{int(h[:6], 16) % 5000:04d}", "merchant_id": f"MER-{meta['mcc']}-{meta['merchant_city']}",
                "merchant_name": f"{meta['merchant_city']} {meta['mcc']}", "device_id": f"DEV-D{int(h[6:12], 16) % 8000:04d}",
                "ip": None, "fraud_label": row["fraud_label"], "fraud_type": "dataset_fraud" if row["fraud_label"] else "genuine",
                "ring_id": None, **meta,
                "transaction": {"transaction_id": row["transaction_id"], "prompt": row["prompt"]}}

    def burst(self, n: int = 20, ftype: Optional[str] = None) -> List[dict]:
        fraud = [r for r in self.rows if r["fraud_label"]] or self.rows
        keep, self.rows = self.rows, fraud
        try:
            return [self.next() for _ in range(n)]
        finally:
            self.rows = keep

    def __iter__(self):
        while True:
            yield self.next()


class MixSource:
    """Labelled dataset rows (real distribution, real labels) interleaved with synthetic traffic (rings, devices, IPs)."""

    def __init__(self, dataset: DatasetSource, synthetic: SyntheticSource, dataset_share: float = 0.5):
        self.d, self.s, self.share = dataset, synthetic, dataset_share
        self.rnd = random.Random()

    def next(self) -> dict:
        return self.d.next() if self.rnd.random() < self.share else self.s.next()

    def burst(self, n: int = 20, ftype: Optional[str] = None) -> List[dict]:
        return self.s.burst(n, ftype)

    def __iter__(self):
        while True:
            yield self.next()


def make_source(kind: Optional[str] = None, fraud_rate: Optional[float] = None, seed: Optional[int] = None):
    """kind: synthetic | dataset | mix | auto. dataset/mix/auto need DATASET_URI; without it (or if it is not
    published yet) they fall back to synthetic so the stream never stops."""
    kind = (kind or os.getenv("PRODUCER_SOURCE", "auto")).lower()
    fraud_rate = float(os.getenv("FRAUD_RATE", "0.04")) if fraud_rate is None else fraud_rate
    uri = os.getenv("DATASET_URI", "")
    synthetic = SyntheticSource(fraud_rate=fraud_rate, seed=seed)
    if kind in ("dataset", "auto", "mix") and uri:
        try:
            ds = DatasetSource(uri, seed)
            return MixSource(ds, synthetic, float(os.getenv("DATASET_SHARE", "0.5"))) if kind == "mix" else ds
        except Exception as e:  # noqa: BLE001 - fall back to synthetic, keep the stream alive
            if kind == "dataset":
                raise
            logger.warning("Dataset %s not usable (%s) - using synthetic transactions", uri, e)
    return synthetic
