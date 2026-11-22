"""Stream service: generator, champion/fallback scoring, store analytics, API, and the Kinesis bus (moto, no network)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sentinel_stream import action_for  # noqa: E402
from sentinel_stream.generator import FRAUD_TYPES, SyntheticSource, prompt_meta  # noqa: E402
from sentinel_stream.processor import Metrics, score_batch  # noqa: E402
from sentinel_stream.store import Store  # noqa: E402


class FakeClient:
    def __init__(self, name, decision="ALLOW", p=0.01, up=True):
        self.name, self.decision, self.p, self.up = name, decision, p, up
        self.calls = 0

    def ready(self):
        return self.up

    def score(self, items):
        self.calls += 1
        if not self.up:
            return None, 1.0
        return [{"transaction_id": t.get("transaction_id"), "fraud_probability": self.p, "decision": self.decision,
                 "confidence": 0.9, "reason_codes": ["R01_NEW_DEVICE"] if self.decision != "ALLOW" else [],
                 "explanation": "x"} for t in items], 12.0


def test_synthetic_events_have_ground_truth_and_scoreable_fields():
    g = SyntheticSource(fraud_rate=0.5, seed=7)
    evs = [g.next() for _ in range(200)]
    assert {e["fraud_label"] for e in evs} == {0, 1}
    for e in evs:
        tx = e["transaction"]
        assert tx["amount"] > 0 and 0 <= tx["hour"] <= 23 and e["customer_id"].startswith("CUST-")
        assert e["fraud_type"] in FRAUD_TYPES + ("genuine",)
    burst = g.burst(10, "card_testing")
    assert len(burst) == 10 and all(e["fraud_label"] == 1 and e["fraud_type"] == "card_testing" for e in burst)
    assert len({e["ring_id"] for e in burst}) == 1          # one ring per attack wave


def test_fraud_patterns_look_risky():
    g = SyntheticSource(seed=1)
    ato = g.fraud("account_takeover")["transaction"]
    assert ato["new_device"] and ato["new_city"] and ato["fails_24h"] >= 2
    mule = g.fraud("mule_account")["transaction"]
    assert mule["account_age_days"] <= 20 and mule["avg_90d"] is None


def test_prompt_meta():
    m = prompt_meta("[TXN] id=X amt=74999.0 mcc=electronics channel=card time=03h Sun\n"
                    "[CTX] new_device=yes city=Patna merchant_city=Mumbai new_city=yes")
    assert m == {"amount": 74999.0, "mcc": "electronics", "channel": "card", "hour": 3, "city": "Patna",
                 "merchant_city": "Mumbai"}


def test_champion_decides_and_shadows_are_recorded():
    evs = [SyntheticSource(seed=2).next() for _ in range(3)]
    clients = {"xgboost": FakeClient("xgboost", "BLOCK", 0.97), "lora": FakeClient("lora", "REVIEW", 0.5),
               "qlora": FakeClient("qlora", up=False)}
    recs = score_batch(evs, clients, "xgboost")
    assert [r["decision"] for r in recs] == ["BLOCK"] * 3
    assert recs[0]["champion"] == "xgboost" and recs[0]["lora_decision"] == "REVIEW"
    assert recs[0]["action"] == "BLOCK_AND_ALERT" and recs[0].get("qlora_p") is None
    assert recs[0]["xgboost_ms"] == 4.0                        # per-transaction share of the batch round trip


def test_fallback_when_champion_down_and_fail_closed():
    ev = [SyntheticSource(seed=3).next()]
    recs = score_batch(ev, {"xgboost": FakeClient("xgboost", up=False), "lora": FakeClient("lora", "ALLOW")}, "xgboost")
    assert recs[0]["champion"] == "lora" and recs[0]["decision"] == "ALLOW"
    recs = score_batch(ev, {"xgboost": FakeClient("xgboost", up=False)}, "xgboost")
    assert recs[0]["decision"] == "REVIEW" and recs[0]["champion"] == "none"      # never auto-approve


def test_action_text_for_non_fraud():
    assert action_for("ALLOW")["text"].startswith("Non-fraud")
    assert action_for("BLOCK")["action"] == "BLOCK_AND_ALERT"


def _filled_store(n=300):
    st = Store(":memory:")
    g = SyntheticSource(fraud_rate=0.3, seed=5)
    evs = [g.next() for _ in range(n)]
    clients = {"xgboost": FakeClient("xgboost", "ALLOW"), "lora": FakeClient("lora", "ALLOW")}
    recs = score_batch(evs, clients)
    for r, e in zip(recs, evs):                                  # make decisions follow the label for the test
        if e["fraud_label"]:
            r["decision"] = r["xgboost_decision"] = "BLOCK"
            r["reason_codes"] = ["R01_NEW_DEVICE", "R05_AMOUNT_SPIKE"]
    st.insert(recs)
    st.insert(recs[:10])                                         # duplicates (at-least-once redelivery) are ignored
    return st, evs


def test_store_summary_groups_and_model_quality():
    st, evs = _filled_store()
    s = st.summary(60)
    assert s["totals"]["n"] == len(evs)
    assert s["totals"]["block"] == sum(e["fraud_label"] for e in evs)
    assert s["models"]["xgboost"]["recall"] == 1.0 and s["models"]["xgboost"]["precision"] == 1.0
    assert s["models"]["lora"]["recall"] == 0.0
    assert s["by_device"] and s["by_device"][0]["key"].startswith("DEV-ATK")   # fraud rings surface on top
    assert s["by_ring"] and s["reasons"]
    assert st.totals()["events"] == len(evs) and len(st.recent(5)) == 5


def test_metrics_counts():
    m = Metrics("")
    m.add([{"decision": "BLOCK", "fraud_label": 1}, {"decision": "ALLOW", "fraud_label": 1, "xgboost_ms": 3.0}])
    assert m.c["Blocked"] == 1 and m.c["FraudMissed"] == 1 and m.lat["xgboost"] == [3.0]
    m.flush(force=True)
    assert m.c["Transactions"] == 0


def test_api_without_streaming(monkeypatch, tmp_path):
    monkeypatch.delenv("KINESIS_ENABLED", raising=False)
    monkeypatch.setenv("STORE_PATH", str(tmp_path / "s.db"))
    monkeypatch.setenv("CONTROL_API_KEY", "ck")
    from fastapi.testclient import TestClient

    from sentinel_stream import api
    api.S.clear()
    st, _ = _filled_store(50)
    api.S["store"] = st
    with TestClient(api.app) as c:
        assert c.get("/live/health").json()["status"] == "ok"
        assert "Live transaction flow" in c.get("/live/flow").text
        f = c.get("/live/api/flow?since=0").json()
        assert f["head"] == 50 and len(f["events"]) == 8 and f["events"][-1]["id"] == 50
        assert c.get("/live/api/flow?since=45").json()["events"][0]["id"] == 46
        assert c.get("/live/api/summary?minutes=30").json()["totals"]["n"] == 50
        assert len(c.get("/live/api/events?limit=7").json()["events"]) == 7
        assert c.get("/live/api/system").json()["mem_total_mb"] > 0
        assert c.post("/live/api/inject", json={"n": 5}).status_code == 401
        assert c.post("/live/api/inject", json={"n": 5}, headers={"X-Control-Key": "ck"}).status_code == 503
        assert c.post("/live/api/producer", json={"action": "start"}).status_code == 401
        assert c.get("/live/api/producer").json() == {"running": False}
        assert f["pipeline"]["producer"]["running"] is False


@pytest.mark.parametrize("ftype", FRAUD_TYPES)
def test_every_fraud_type_generates(ftype):
    e = SyntheticSource(seed=9).fraud(ftype)
    assert e["fraud_label"] == 1 and e["fraud_type"] == ftype and e["device_id"].startswith("DEV-ATK")


def test_mix_source_replays_dataset_and_falls_back(tmp_path, monkeypatch):
    import gzip
    import json

    from sentinel_stream.generator import DatasetSource, MixSource, make_source

    d = tmp_path / "holdout"
    d.mkdir()
    prompt = ("[TXN] id=TX_1 amt=950.0 mcc=food channel=UPI time=13h Tue\n[CUST] age=30\n"
              "[CTX] new_device=no city=Pune merchant_city=Pune new_city=no")
    with gzip.open(d / "part-00001.jsonl.gz", "wt", encoding="utf-8") as f:
        for i in range(10):
            f.write(json.dumps({"transaction_id": f"TX_{i}", "prompt": prompt, "fraud_label": int(i == 0)}) + "\n")
    monkeypatch.setenv("DATASET_URI", str(d))
    src = make_source("mix", seed=1)
    assert isinstance(src, MixSource)
    evs = [src.next() for _ in range(60)]
    assert {e["source"] for e in evs} == {"dataset", "synthetic"}
    ds = next(e for e in evs if e["source"] == "dataset")
    assert ds["transaction"]["prompt"] == prompt and ds["mcc"] == "food" and ds["amount"] == 950.0
    assert isinstance(make_source("dataset"), DatasetSource)
    monkeypatch.setenv("DATASET_URI", str(tmp_path / "missing"))
    assert type(make_source("mix")).__name__ == "SyntheticSource"     # not published yet -> keep streaming


class FakeProducer:
    def __init__(self):
        self.msgs = []

    def produce(self, topic, key=None, value=None):
        self.msgs.append((topic, key, value))

    def poll(self, t):
        return 0

    def flush(self, t=None):
        return 0


def test_traffic_starts_only_on_demand_and_stops():
    import time

    from sentinel_stream.producer import Traffic

    fp = FakeProducer()
    t = Traffic(producer=fp, source=SyntheticSource(seed=1))
    assert not t.running and t.status()["sent"] == 0          # auto mode is OFF until the button is pressed
    t.start(rate=50)
    deadline = time.time() + 5
    while t.sent < 5 and time.time() < deadline:
        time.sleep(0.05)
    t.stop()
    assert not t.running and t.sent >= 5 and len(fp.msgs) == t.sent
    assert fp.msgs[0][0] == "sentinel-transactions"
    n = t.burst(7, "card_testing")
    assert n == 7 and t.attacks == 1
    assert t.start(rate=999)["rate"] == 50.0                  # clamped
    t.stop()


# ------------------------------------------------------------------ Amazon Kinesis (moto: in-memory AWS)
@pytest.fixture
def kinesis(monkeypatch):
    moto = pytest.importorskip("moto")
    for k, v in {"AWS_ACCESS_KEY_ID": "test", "AWS_SECRET_ACCESS_KEY": "test", "AWS_DEFAULT_REGION": "ap-south-1",
                 "AWS_REGION": "ap-south-1"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("KINESIS_ENDPOINT", raising=False)
    with moto.mock_aws():
        from sentinel_stream.bus import ensure_streams, kinesis_client

        c = kinesis_client()
        ensure_streams(c, ["sentinel-transactions", "sentinel-decisions"])
        yield c


def _drain(reader, want, seconds=10):
    import time

    out, t0 = [], time.time()
    while len(out) < want and time.time() - t0 < seconds:
        recs, _ = reader.read()
        out += recs
        time.sleep(0.05)
    return out


def test_kinesis_producer_and_reader_round_trip(kinesis):
    import time

    from sentinel_stream.bus import KinesisProducer, ShardReader

    p = KinesisProducer(kinesis)
    g = SyntheticSource(seed=4)
    evs = [g.next() for _ in range(620)]                       # > 500: PutRecords is split into two calls
    for e in evs:
        p.produce("sentinel-transactions", key=e["customer_id"].encode(), value=e)
    assert p.poll(0) == 620 and p.sent == 620 and not p.buf    # buffer >= 500 -> sent on poll
    got = _drain(ShardReader("sentinel-transactions", kinesis, start="TRIM_HORIZON", limit=1000), 620)
    assert [e["event_id"] for e in got] == [e["event_id"] for e in evs]
    assert _drain(ShardReader("sentinel-transactions", kinesis, start="LATEST"), 1, seconds=1) == []
    assert len(_drain(ShardReader("sentinel-transactions", kinesis, start=time.time() - 60, limit=1000), 620)) == 620


def test_processor_scores_from_kinesis_and_rebuilds_after_restart(kinesis):
    import time

    from sentinel_stream.bus import KinesisProducer, ShardReader
    from sentinel_stream.processor import Processor
    from sentinel_stream.producer import Traffic

    clients = {"xgboost": FakeClient("xgboost", "BLOCK", 0.97), "lora": FakeClient("lora", "REVIEW", 0.5)}
    st = Store(":memory:")
    proc = Processor(st, clients=clients, kinesis=kinesis)
    proc.start()
    deadline = time.time() + 10
    while proc.state["stream"] != "connected" and time.time() < deadline:
        time.sleep(0.05)
    t = Traffic(producer=KinesisProducer(kinesis), source=SyntheticSource(seed=8))
    assert t.burst(12, "account_takeover") == 12               # button "Inject attack" -> Kinesis
    while st.max_id() < 12 and time.time() < deadline:
        time.sleep(0.1)
    proc.stop()
    proc.join(5)
    assert st.totals()["events"] == 12 and proc.state["consumed"] == 12
    recs = st.recent(12)
    assert {r["decision"] for r in recs} == {"BLOCK"} and {r["action"] for r in recs} == {"BLOCK_AND_ALERT"}
    out = _drain(ShardReader("sentinel-decisions", kinesis, start="TRIM_HORIZON"), 12)   # downstream consumers
    assert len(out) == 12 and out[0]["action"] == "BLOCK_AND_ALERT" and out[0]["action_text"].startswith("Fraud")

    fresh = Store(":memory:")                                  # new Fargate task: empty disk
    assert Processor(fresh, clients=clients, kinesis=kinesis).rebuild() == 12
    assert fresh.totals()["events"] == 12 and fresh.last_event_ts() > 0
