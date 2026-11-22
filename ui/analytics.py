"""Live Flow + Analytics tabs: Kinesis stream analytics (who sends what, decisions, models) and infrastructure
metrics (CloudWatch Container Insights on AWS, host gauges from the stream service everywhere).

Env:
  STREAM_URL         server-side URL of the stream service       default http://localhost:8003
  PUBLIC_STREAM_URL  browser URL of the flow page (iframe)        default STREAM_URL/live  (AWS: /live via the ALB)
  CONTROL_API_KEY    lets the dashboard inject attack waves into Kinesis
  ECS_CLUSTER, ECS_SERVICES (comma list), ASG_NAME, AWS_REGION, CW_NAMESPACE   -> CloudWatch panels
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import List

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components

from style import DECISION_COLOR, GRAY, LORA_COLOR, QLORA_COLOR, XGB_COLOR, esc, kpi, pill, plotly_layout

STREAM_URL = os.getenv("STREAM_URL", "http://localhost:8003").rstrip("/")
PUBLIC_STREAM_URL = os.getenv("PUBLIC_STREAM_URL", STREAM_URL + "/live").rstrip("/")
MODEL_COLOR = {"xgboost": XGB_COLOR, "lora": LORA_COLOR, "qlora": QLORA_COLOR}
MODEL_NAME = {"xgboost": "XGBoost", "lora": "LoRA", "qlora": "QLoRA"}
DEC = ["ALLOW", "REVIEW", "BLOCK"]


def _get(path: str, timeout: float = 8, **params):
    try:
        r = requests.get(f"{STREAM_URL}{path}", params=params, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except (requests.RequestException, ValueError) as e:
        return None, f"{type(e).__name__}: {e}"


@st.cache_data(ttl=5, show_spinner=False)
def summary(minutes: int):
    return _get("/live/api/summary", minutes=minutes)


@st.cache_data(ttl=5, show_spinner=False)
def system():
    return _get("/live/api/system")


def pct(v):
    return "—" if v is None else f"{100 * v:.1f}%"


def money(v):
    return f"₹{v:,.0f}" if v is not None else "—"


# ------------------------------------------------------------------ Live Flow
def _control(path: str, body: dict):
    key = os.getenv("CONTROL_API_KEY", "")
    try:
        r = requests.post(f"{STREAM_URL}{path}", headers={"X-Control-Key": key}, json=body, timeout=15)
        return (r.json(), None) if r.ok else (None, f"HTTP {r.status_code}: {r.text[:200]}")
    except (requests.RequestException, ValueError) as e:
        return None, str(e)


def render_flow():
    st.markdown('<div class="card"><h4>Amazon Kinesis → stream scorer → models on Hyperstack → decision → action</h4>'
                '<div class="muted">Press <b>Start auto traffic</b>: the producer publishes transactions to Kinesis, the '
                'scorer consumes them, XGBoost (champion) decides while LoRA and QLoRA score the same transactions in '
                'shadow mode, and every decision triggers an action: <b>ALLOW</b> = non-fraud, approved, no further '
                'action · <b>REVIEW</b> = OTP step-up + analyst queue · <b>BLOCK</b> = blocked, card frozen, alert '
                'published.</div></div>', unsafe_allow_html=True)
    if os.getenv("CONTROL_API_KEY"):
        status, err = _get("/live/api/producer")
        running = bool(status and status.get("running"))
        c = st.columns([1.3, 1.3, 2, 2, 1.2, 1.4])
        c[0].markdown(pill("AUTO TRAFFIC ON", "p-allow") if running else pill("AUTO TRAFFIC OFF", "p-gray"),
                      unsafe_allow_html=True)
        c[0].caption(f"{(status or {}).get('sent', 0):,} sent" if status else (err or "")[:80])
        rate = c[2].slider("Transactions / second", 0.5, 20.0, float((status or {}).get("rate") or 2.0), 0.5)
        if c[1].button("Stop" if running else "Start auto traffic", type="primary", use_container_width=True):
            res, e = _control("/live/api/producer", {"action": "stop" if running else "start", "rate": rate})
            st.error(e) if e else st.rerun()
        if running and abs(rate - float(status.get("rate") or rate)) > 1e-9:
            _control("/live/api/producer", {"action": "start", "rate": rate})
        kind = c[3].selectbox("Attack wave", ["mixed", "account_takeover", "card_testing", "mule_account",
                                              "impossible_travel"])
        n = c[4].number_input("Size", 5, 100, 25, step=5)
        c[5].markdown("<br>", unsafe_allow_html=True)
        if c[5].button("Inject attack", use_container_width=True):
            res, e = _control("/live/api/inject", {"n": int(n), **({} if kind == "mixed" else {"fraud_type": kind})})
            st.error(e) if e else st.success(f"Sent {res.get('sent')} fraud transactions to Kinesis")
        if status and status.get("error"):
            st.error(f"Producer error: {status['error']}")
    components.iframe(f"{PUBLIC_STREAM_URL}/flow", height=900, scrolling=True)


# ------------------------------------------------------------------ CloudWatch
@st.cache_data(ttl=60, show_spinner=False)
def cloudwatch(hours: int):
    cluster, region = os.getenv("ECS_CLUSTER"), os.getenv("AWS_REGION", "ap-south-1")
    services = [s for s in os.getenv("ECS_SERVICES", "").split(",") if s]
    if not cluster or not services:
        return None, "CloudWatch panels appear when the dashboard runs on AWS (ECS_CLUSTER / ECS_SERVICES set)."
    try:
        import boto3

        cw = boto3.client("cloudwatch", region_name=region)
        queries = []
        for i, svc in enumerate(services):
            for metric in ("CpuUtilized", "MemoryUtilized"):
                queries.append({"Id": f"{metric[:3].lower()}{i}", "Label": f"{svc}|{metric}", "MetricStat": {
                    "Metric": {"Namespace": "ECS/ContainerInsights", "MetricName": metric,
                               "Dimensions": [{"Name": "ClusterName", "Value": cluster}, {"Name": "ServiceName", "Value": svc}]},
                    "Period": 60, "Stat": "Average"}})
        if os.getenv("ASG_NAME"):
            queries.append({"Id": "ec2cpu", "Label": "instance|CPUUtilization", "MetricStat": {
                "Metric": {"Namespace": "AWS/EC2", "MetricName": "CPUUtilization",
                           "Dimensions": [{"Name": "AutoScalingGroupName", "Value": os.getenv("ASG_NAME")}]},
                "Period": 60, "Stat": "Average"}})
        ns = os.getenv("CW_NAMESPACE", "")
        if ns:
            for m in ("Transactions", "Blocked", "Reviewed", "StreamLagMs"):
                queries.append({"Id": f"app{m.lower()}", "Label": f"app|{m}", "MetricStat": {
                    "Metric": {"Namespace": ns, "MetricName": m, "Dimensions": [{"Name": "Service", "Value": "stream"}]},
                    "Period": 60, "Stat": "Sum" if m != "StreamLagMs" else "Maximum"}})
        end = datetime.now(timezone.utc)
        rows = []
        for i in range(0, len(queries), 500):
            res = cw.get_metric_data(MetricDataQueries=queries[i:i + 500], StartTime=end - timedelta(hours=hours), EndTime=end)
            for r in res["MetricDataResults"]:
                who, metric = r["Label"].split("|")
                rows += [{"service": who, "metric": metric, "t": t, "v": v} for t, v in zip(r["Timestamps"], r["Values"])]
        return pd.DataFrame(rows), None
    except Exception as e:  # noqa: BLE001 - no credentials locally, missing permissions, ...
        return None, f"CloudWatch unavailable: {type(e).__name__}: {e}"


def _bar(rows: List[dict], title: str, color: str, label="key", height=300):
    if not rows:
        st.caption(f"{title}: no data yet")
        return
    df = pd.DataFrame(rows)[::-1]
    fig = go.Figure()
    for col, name, c in (("block", "BLOCK", DECISION_COLOR["BLOCK"]), ("review", "REVIEW", DECISION_COLOR["REVIEW"])):
        fig.add_trace(go.Bar(y=df[label].astype(str), x=df[col], name=name, orientation="h", marker_color=c))
    st.plotly_chart(plotly_layout(fig, height=height, title=title, barmode="stack"), use_container_width=True)


# ------------------------------------------------------------------ Analytics
def render_analytics():
    top = st.columns([2, 1, 1])
    window = top[0].radio("Window", ["15 min", "1 hour", "6 hours", "24 hours", "7 days"], index=1, horizontal=True)
    minutes = {"15 min": 15, "1 hour": 60, "6 hours": 360, "24 hours": 1440, "7 days": 10080}[window]
    if top[2].button("Refresh analytics", use_container_width=True):
        summary.clear(); system.clear(); cloudwatch.clear()
    data, err = summary(minutes)
    if err:
        st.info(f"Stream service not reachable at {STREAM_URL} ({err}). Start the stream service "
                "(`docker compose up -d`) or deploy to AWS.")
        return
    t = data["totals"]
    n = t.get("n") or 0
    k = st.columns(6)
    k[0].markdown(kpi("Transactions", f"{n:,}", f"last {window}"), unsafe_allow_html=True)
    k[1].markdown(kpi("Approved", f"{t.get('allow', 0):,}", pct(t.get("allow", 0) / n if n else None)), unsafe_allow_html=True)
    k[2].markdown(kpi("Review", f"{t.get('review', 0):,}", pct(t.get("review", 0) / n if n else None)), unsafe_allow_html=True)
    k[3].markdown(kpi("Blocked", f"{t.get('block', 0):,}", pct(t.get("block", 0) / n if n else None)), unsafe_allow_html=True)
    k[4].markdown(kpi("Fraud stopped", money(t.get("fraud_amount_stopped")), f"missed {money(t.get('fraud_amount_missed'))}"),
                  unsafe_allow_html=True)
    k[5].markdown(kpi("End-to-end", f"{t.get('e2e_ms') or 0:.0f} ms", "Kinesis in → decision"), unsafe_allow_html=True)

    ts = pd.DataFrame(data["timeseries"])
    if not ts.empty:
        ts["time"] = pd.to_datetime(ts["t"], unit="s")
        fig = go.Figure()
        for d in DEC:
            s = ts[ts["decision"] == d]
            fig.add_trace(go.Scatter(x=s["time"], y=s["n"], name=d, stackgroup="one", mode="lines",
                                     line=dict(width=0.5, color=DECISION_COLOR[d])))
        st.plotly_chart(plotly_layout(fig, height=300, title=f"Decisions per {data['bucket_s'] // 60} min"),
                        use_container_width=True)

    st.markdown('<div class="section">Who is sending risky transactions</div>', unsafe_allow_html=True)
    c = st.columns(3)
    with c[0]:
        _bar(data["by_device"], "Devices (REVIEW + BLOCK)", XGB_COLOR)
    with c[1]:
        _bar(data["by_customer"], "Customers (REVIEW + BLOCK)", LORA_COLOR)
    with c[2]:
        _bar(data["by_merchant"], "Merchants (REVIEW + BLOCK)", QLORA_COLOR)
    c = st.columns(3)
    with c[0]:
        _bar(data["by_ip"], "Source IPs (REVIEW + BLOCK)", XGB_COLOR)
    with c[1]:
        _bar(data["by_ring"], "Fraud rings (ground truth)", XGB_COLOR)
    with c[2]:
        rs = data.get("reasons") or {}
        if rs:
            items = list(rs.items())[:8][::-1]
            fig = go.Figure(go.Bar(x=[v for _, v in items], y=[k.split("_", 1)[1].replace("_", " ").title() for k, _ in items],
                                   orientation="h", marker_color=LORA_COLOR))
            st.plotly_chart(plotly_layout(fig, height=300, title="Reason codes on flagged transactions"), use_container_width=True)

    st.markdown('<div class="section">What kind of transactions</div>', unsafe_allow_html=True)
    c = st.columns(3)
    for col, key, title in zip(c, ("by_mcc", "by_channel", "by_fraud_type"), ("Merchant category", "Channel", "Fraud type (truth)")):
        with col:
            df = pd.DataFrame(data[key])
            if df.empty:
                continue
            df["allow"] = df["n"] - df["review"] - df["block"]
            fig = go.Figure()
            for d, colname in (("ALLOW", "allow"), ("REVIEW", "review"), ("BLOCK", "block")):
                fig.add_trace(go.Bar(x=df["key"].astype(str), y=df[colname], name=d, marker_color=DECISION_COLOR[d]))
            st.plotly_chart(plotly_layout(fig, height=300, title=title, barmode="stack"), use_container_width=True)
    c = st.columns(2)
    with c[0]:
        hr = pd.DataFrame(data["by_hour"])
        hr = hr.dropna(subset=["key"]) if not hr.empty else hr
        if not hr.empty:
            pv = hr.pivot_table(index="decision", columns="key", values="n", aggfunc="sum").reindex(DEC).fillna(0)
            fig = go.Figure(go.Heatmap(z=pv.values, x=[f"{int(h):02d}h" for h in pv.columns], y=pv.index,
                                       colorscale="Blues", showscale=False))
            st.plotly_chart(plotly_layout(fig, height=260, title="Decisions by hour of day"), use_container_width=True)
    with c[1]:
        city = pd.DataFrame(data["by_city"])
        if not city.empty:
            city["risk"] = (city["block"] + city["review"]) / city["n"]
            fig = go.Figure(go.Bar(x=city["key"], y=city["risk"], marker_color=DECISION_COLOR["REVIEW"]))
            st.plotly_chart(plotly_layout(fig, height=260, title="Flag rate by customer city", yaxis_tickformat=".0%"),
                            use_container_width=True)

    st.markdown('<div class="section">Models on the live stream</div>', unsafe_allow_html=True)
    models = data["models"]
    pipe = data.get("pipeline", {}).get("models", {})
    c = st.columns(3)
    for col, m in zip(c, ("xgboost", "lora", "qlora")):
        s, p = models.get(m, {}), pipe.get(m, {})
        role = p.get("role", "")
        status = pill("online", "p-allow") if p.get("healthy") else pill(p.get("status") or "offline", "p-gray")
        col.markdown(
            f'<div class="card"><h4><span class="dot" style="background:{MODEL_COLOR[m]}"></span>{MODEL_NAME[m]} '
            f'<span class="muted">· {esc(role)}</span> {status}</h4>'
            f'<div class="muted">scored {s.get("scored", 0):,} · avg {s.get("avg_ms") or 0:.1f} ms · p95 {s.get("p95_ms") or 0:.1f} ms / txn</div>'
            f'<div>Live recall <b>{pct(s.get("recall"))}</b> · precision <b>{pct(s.get("precision"))}</b> · '
            f'precision@BLOCK <b>{pct(s.get("precision_block"))}</b> · agrees with decision <b>{pct(s.get("agreement"))}</b></div>'
            f'</div>', unsafe_allow_html=True)
    st.caption("Live recall/precision use the ground-truth label carried by each streamed transaction "
               "(synthetic fraud rings or the labelled holdout dataset).")

    st.markdown('<div class="section">Infrastructure</div>', unsafe_allow_html=True)
    sysd, _ = system()
    if sysd:
        k = st.columns(4)
        k[0].markdown(kpi("Task CPU", f"{sysd['cpu_percent']:.0f}%", f"{sysd['cpus']} vCPU"), unsafe_allow_html=True)
        k[1].markdown(kpi("Task memory", f"{sysd['mem_percent']:.0f}%", f"{sysd['mem_used_mb']:,} / {sysd['mem_total_mb']:,} MB"),
                      unsafe_allow_html=True)
        k[2].markdown(kpi("Disk", f"{sysd['disk_percent']:.0f}%", "event store"), unsafe_allow_html=True)
        la = sysd.get("load_avg")
        k[3].markdown(kpi("Load average", f"{la[0]:.2f}" if la else "—", "1 min"), unsafe_allow_html=True)
    hours = st.radio("CloudWatch range", [1, 3, 12, 24], index=1, horizontal=True, format_func=lambda h: f"{h} h")
    df, msg = cloudwatch(hours)
    if df is None or df.empty:
        st.caption(msg or "No CloudWatch datapoints yet (Container Insights publishes every minute).")
        return
    c = st.columns(2)
    for col, metric, title, unit in ((c[0], "CpuUtilized", "CPU (Fargate task)", "CPU units (1024 = 1 vCPU)"),
                                     (c[1], "MemoryUtilized", "Memory (Fargate task)", "MB")):
        with col:
            fig = go.Figure()
            for svc, g in df[df.metric == metric].groupby("service"):
                g = g.sort_values("t")
                fig.add_trace(go.Scatter(x=g["t"], y=g["v"], name=svc.split("-")[-1], mode="lines"))
            st.plotly_chart(plotly_layout(fig, height=300, title=title, yaxis_title=unit), use_container_width=True)
    c = st.columns(2)
    with c[0]:
        g = df[df.metric == "CPUUtilization"].sort_values("t")
        if not g.empty:
            fig = go.Figure(go.Scatter(x=g["t"], y=g["v"], mode="lines", fill="tozeroy", line=dict(color=GRAY)))
            st.plotly_chart(plotly_layout(fig, height=260, title="EC2 instance CPU %", yaxis_range=[0, 100]),
                            use_container_width=True)
    with c[1]:
        fig = go.Figure()
        for m, col_ in (("Transactions", LORA_COLOR), ("Blocked", DECISION_COLOR["BLOCK"]), ("Reviewed", DECISION_COLOR["REVIEW"])):
            g = df[(df.service == "app") & (df.metric == m)].sort_values("t")
            if not g.empty:
                fig.add_trace(go.Scatter(x=g["t"], y=g["v"], name=m, mode="lines", line=dict(color=col_)))
        if fig.data:
            st.plotly_chart(plotly_layout(fig, height=260, title="Stream metrics (CloudWatch, per minute)"),
                            use_container_width=True)
