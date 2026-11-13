"""SENTINEL dashboard: LLM stats, LLM-vs-XGBoost benchmark, live + batch testing.

    streamlit run ui/app.py
Endpoints come from env (LLM_URL, LLM_API_KEY, XGB_URL, XGB_API_KEY) and can be edited in the sidebar.
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from client import Endpoint  # noqa: E402
from style import (CSS, DECISION_COLOR, GRAY, GREEN, LLM_COLOR, RED, XGB_COLOR, decision_pill,  # noqa: E402
                   dot, esc, kpi, pill, plotly_layout, prob_bar)

st.set_page_config(page_title="SENTINEL", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)

TARGETS = {"recall": 0.85, "precision_block": 0.70, "pr_auc": 0.80, "uplift": 0.03, "p99_ms": 250}
FALLBACK_SCENARIOS = [
    {"name": "Everyday grocery (UPI)", "expect": "ALLOW", "transaction": {
        "transaction_id": "SCN_A", "amount": 640, "mcc": "grocery", "channel": "UPI", "hour": 19, "weekday": "Tue",
        "customer_age": 41, "account_age_days": 1450, "avg_90d": 720, "txns_30d": 14, "fails_24h": 0,
        "new_device": False, "city": "Pune", "merchant_city": "Pune", "new_city": False, "dist_km": "<5",
        "amt_ratio": 0.9, "txns_1h": 0, "amount_24h": 300, "device_users_7d": 1, "history": []}},
    {"name": "3 AM electronics, new device, failed attempts", "expect": "BLOCK", "transaction": {
        "transaction_id": "SCN_B", "amount": 74999, "mcc": "electronics", "channel": "card", "hour": 3, "weekday": "Sun",
        "customer_age": 24, "account_age_days": 420, "avg_90d": 1800, "txns_30d": 6, "fails_24h": 4,
        "new_device": True, "city": "Patna", "merchant_city": "Mumbai", "new_city": True, "dist_km": "1000+",
        "amt_ratio": 41.7, "txns_1h": 3, "amount_24h": 12000, "device_users_7d": 2, "history": []}},
]
MCC = ["grocery", "quick_commerce", "food", "electronics", "gaming", "gift_cards", "healthcare", "travel", "education", "other"]
CHANNELS = ["UPI", "card", "wallet", "net_banking"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DIST = ["<5", "50-200", "200-500", "500-1000", "1000+", "unk"]

ss = st.session_state
ss.setdefault("live_log", [])
ss.setdefault("batch", None)


# ------------------------------------------------------------------ sidebar: connections
with st.sidebar:
    st.markdown('<p class="large-title" style="font-size:24px">Connections</p>', unsafe_allow_html=True)
    st.caption("LLM runs on the Hyperstack GPU · XGBoost on your machine (ngrok)")
    llm_url = st.text_input("LLM URL", os.getenv("LLM_URL", "http://localhost:8000"))
    llm_key = st.text_input("LLM API key", os.getenv("LLM_API_KEY", ""), type="password")
    xgb_url = st.text_input("XGBoost URL", os.getenv("XGB_URL", "http://localhost:8001"))
    xgb_key = st.text_input("XGBoost API key", os.getenv("XGB_API_KEY", ""), type="password")
    if st.button("Refresh", use_container_width=True):
        st.cache_data.clear()

LLM = Endpoint("LLM", llm_url, llm_key)
XGB = Endpoint("XGBoost", xgb_url, xgb_key)


@st.cache_data(ttl=20, show_spinner=False)
def cached_health(url, key, name):
    return Endpoint(name, url, key).health()


@st.cache_data(ttl=120, show_spinner=False)
def cached_stats(url, key, name):
    return Endpoint(name, url, key).stats()


@st.cache_data(ttl=300, show_spinner=False)
def cached_scenarios(url, key):
    ok, data, _ = Endpoint("XGBoost", url, key).scenarios()
    return data if ok and isinstance(data, list) and data else FALLBACK_SCENARIOS


health = {"LLM": cached_health(llm_url, llm_key, "LLM"), "XGBoost": cached_health(xgb_url, xgb_key, "XGBoost")}
stats = {}
for ep in (LLM, XGB):
    ok_h = health[ep.name][0]
    ok_s, data, _ = cached_stats(ep.url, ep.key, ep.name) if ok_h else (False, health[ep.name][1], 0)
    stats[ep.name] = data if ok_s else None
    if ok_h and not ok_s:
        health[ep.name] = (False, data, 0)

with st.sidebar:
    st.markdown('<div class="section">Status</div>', unsafe_allow_html=True)
    for name in ("LLM", "XGBoost"):
        ok, data, ms = health[name]
        if ok:
            dev = data.get("device", "")
            st.markdown(f'<div class="card">{dot(GREEN)}<b>{name}</b> · online · {ms:.0f} ms'
                        f'<div class="muted">{esc(data.get("run", data.get("kind", "")))} {esc(dev)}</div></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="card">{dot(RED)}<b>{name}</b> · offline'
                        f'<div class="muted">{esc(str(data)[:160])}</div></div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
def m_of(name, split="holdout"):
    s = stats.get(name)
    try:
        return s["metrics"][split]["calibrated"]
    except (TypeError, KeyError):
        return None


def fmt(v, digits=3):
    return "—" if v is None else f"{v:.{digits}f}"


def pct(v):
    return "—" if v is None else f"{100 * v:.1f}%"


def status_chip(ok):
    if ok is None:
        return pill("n/a", "p-gray")
    return pill("PASS", "p-allow") if ok else pill("FAIL", "p-block")


def offline_note(name):
    st.info(f"{name} is not reachable. Check its URL / API key in the sidebar.")


st.markdown('<p class="large-title">SENTINEL</p><p class="subtitle">Fraud intelligence LLM · '
            'benchmark vs XGBoost · live testing</p>', unsafe_allow_html=True)
t_over, t_llm, t_bench, t_live, t_batch = st.tabs(["Overview", "LLM", "Benchmark", "Live Test", "Batch Test"])


# ------------------------------------------------------------------ Overview
with t_over:
    cols = st.columns(2)
    for col, name, color in ((cols[0], "LLM", LLM_COLOR), (cols[1], "XGBoost", XGB_COLOR)):
        with col:
            s = stats.get(name)
            m = m_of(name)
            head = esc(s["model"]) if s else "offline"
            st.markdown(f'<div class="card"><h4>{dot(color)}{name}</h4><div class="muted">{head}</div></div>',
                        unsafe_allow_html=True)
            k = st.columns(4)
            k[0].markdown(kpi("PR-AUC", fmt(m and m["pr_auc"]), "holdout"), unsafe_allow_html=True)
            k[1].markdown(kpi("ROC-AUC", fmt(m and m["roc_auc"]), "holdout"), unsafe_allow_html=True)
            k[2].markdown(kpi("Recall", pct(m and m["at_review"]["recall"]), "at REVIEW"), unsafe_allow_html=True)
            k[3].markdown(kpi("Precision", pct(m and m["at_block"]["precision"]), "at BLOCK"), unsafe_allow_html=True)

    st.markdown('<div class="section">Acceptance targets (LLM, holdout)</div>', unsafe_allow_html=True)
    lm, xm = m_of("LLM"), m_of("XGBoost")
    lat = [r["llm_ms"] for r in ss.live_log if r.get("llm_ms") is not None]
    p99 = float(np.percentile(lat, 99)) if len(lat) >= 5 else None
    uplift = (lm["pr_auc"] - xm["pr_auc"]) if lm and xm and lm["pr_auc"] is not None and xm["pr_auc"] is not None else None
    rows = [
        ("OBJ-01", "Recall on fraud (holdout)", "≥ 85%", pct(lm and lm["at_review"]["recall"]),
         None if not lm else lm["at_review"]["recall"] >= TARGETS["recall"]),
        ("OBJ-02", "Precision at BLOCK threshold", "≥ 70%", pct(lm and lm["at_block"]["precision"]),
         None if not lm else lm["at_block"]["precision"] >= TARGETS["precision_block"]),
        ("OBJ-03", "PR-AUC on holdout", "≥ 0.80", fmt(lm and lm["pr_auc"]),
         None if not lm or lm["pr_auc"] is None else lm["pr_auc"] >= TARGETS["pr_auc"]),
        ("OBJ-04", "PR-AUC uplift over XGBoost", "≥ +0.03", "—" if uplift is None else f"{uplift:+.3f}",
         None if uplift is None else uplift >= TARGETS["uplift"]),
        ("OBJ-10", "Scoring latency p99 (live tests, round trip)", "≤ 250 ms",
         "—" if p99 is None else f"{p99:.0f} ms ({len(lat)} calls)", None if p99 is None else p99 <= TARGETS["p99_ms"]),
    ]
    body = "".join(f"<tr><td><b>{a}</b></td><td>{esc(b)}</td><td class='muted'>{esc(c)}</td>"
                   f"<td><b>{esc(d)}</b></td><td>{status_chip(e)}</td></tr>" for a, b, c, d, e in rows)
    st.markdown(f'<div class="card"><table style="width:100%;border-collapse:collapse;font-size:14px">'
                f'<tr class="muted"><td>ID</td><td>Objective</td><td>Target</td><td>Measured</td><td></td></tr>'
                f'{body}</table><div class="muted" style="margin-top:8px">Latency fills in as you run '
                f'Live Test calls (≥5 needed).</div></div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ LLM
with t_llm:
    s = stats.get("LLM")
    if not s:
        offline_note("LLM")
    else:
        info = s.get("info", {})
        c = st.columns(4)
        c[0].markdown(kpi("Base model", str(info.get("base_model", "—")).split("/")[-1], str(info.get("method", "")).upper()), unsafe_allow_html=True)
        c[1].markdown(kpi("Device", str(info.get("device", "—"))[:22], str(info.get("host", ""))), unsafe_allow_html=True)
        c[2].markdown(kpi("Train time", f"{(info.get('train_runtime_s') or 0) / 60:.1f} min", f"{info.get('train_rows', '—')} rows"), unsafe_allow_html=True)
        c[3].markdown(kpi("Final train loss", fmt(info.get("train_loss"), 4), s.get("run", "")), unsafe_allow_html=True)

        left, right = st.columns([3, 2])
        with left:
            lc = s.get("loss_curve") or []
            if lc:
                df = pd.DataFrame(lc)
                fig = go.Figure(go.Scatter(x=df["step"], y=df["loss"], mode="lines", line=dict(color=LLM_COLOR, width=2.5)))
                fig.add_trace(go.Scatter(x=df["step"], y=df["loss"].rolling(10, min_periods=1).mean(), mode="lines",
                                         name="smoothed", line=dict(color="#1C1C1E", width=1.5, dash="dot")))
                st.plotly_chart(plotly_layout(fig, title="Training loss", showlegend=False), use_container_width=True)
        with right:
            cal = s.get("calibration", {})
            st.markdown(f'<div class="card"><h4>Decision thresholds</h4>'
                        f'<div class="muted">Platt-calibrated on validation, frozen for holdout + serving</div><br>'
                        f'{decision_pill("REVIEW")} p ≥ <b>{fmt(cal.get("review_threshold"), 4)}</b> '
                        f'<span class="muted">(recall ≥ {pct(cal.get("target_recall"))})</span><br>'
                        f'{decision_pill("BLOCK")} p ≥ <b>{fmt(cal.get("block_threshold"), 4)}</b> '
                        f'<span class="muted">(precision ≥ {pct(cal.get("target_precision"))})</span></div>',
                        unsafe_allow_html=True)
            rows = []
            for split in ("validation", "holdout"):
                m = m_of("LLM", split)
                if m:
                    rows.append({"split": split, "rows": m["rows"], "fraud": m["fraud_rows"], "PR-AUC": m["pr_auc"],
                                 "ROC-AUC": m["roc_auc"], "ECE": m["ece"], "recall@REVIEW": m["at_review"]["recall"],
                                 "precision@BLOCK": m["at_block"]["precision"]})
            if rows:
                st.dataframe(pd.DataFrame(rows).set_index("split").round(4), use_container_width=True)
        try:
            per = s["metrics"]["holdout"]["reasons"]["per_code_f1"]
            fig = go.Figure(go.Bar(x=list(per.values()), y=[k.split("_", 1)[1].replace("_", " ").title() for k in per],
                                   orientation="h", marker_color=LLM_COLOR))
            st.plotly_chart(plotly_layout(fig, height=300, title="Reason-code F1 (holdout)", xaxis_range=[0, 1]),
                            use_container_width=True)
        except (KeyError, TypeError):
            pass


# ------------------------------------------------------------------ Benchmark
with t_bench:
    split = st.radio("Split", ["holdout", "validation"], horizontal=True, label_visibility="collapsed") or "holdout"
    lm, xm = m_of("LLM", split), m_of("XGBoost", split)
    if not lm and not xm:
        offline_note("Neither model")
    else:
        def row(m):
            if not m:
                return [None] * 8
            return [m["pr_auc"], m["roc_auc"], m["ece"], m["at_review"]["recall"], m["at_review"]["precision"],
                    m["at_block"]["recall"], m["at_block"]["precision"], m["at_review"]["false_positive_rate"]]
        names = ["PR-AUC", "ROC-AUC", "ECE (↓)", "Recall@REVIEW", "Precision@REVIEW", "Recall@BLOCK",
                 "Precision@BLOCK", "FPR@REVIEW (↓)"]
        df = pd.DataFrame({"LLM": row(lm), "XGBoost": row(xm)}, index=names)
        df["Δ LLM − XGB"] = df["LLM"] - df["XGBoost"]
        k = st.columns(3)
        k[0].markdown(kpi("LLM PR-AUC", fmt(lm and lm["pr_auc"]), split), unsafe_allow_html=True)
        k[1].markdown(kpi("XGBoost PR-AUC", fmt(xm and xm["pr_auc"]), split), unsafe_allow_html=True)
        d = df.loc["PR-AUC", "Δ LLM − XGB"]
        k[2].markdown(kpi("Uplift", "—" if pd.isna(d) else f"{d:+.3f}", "target ≥ +0.03"), unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            for name, m, color in (("LLM", lm, LLM_COLOR), ("XGBoost", xm, XGB_COLOR)):
                cv = (m or {}).get("curves", {}).get("pr")
                if cv:
                    fig.add_trace(go.Scatter(x=cv["recall"], y=cv["precision"], mode="lines", name=name,
                                             line=dict(color=color, width=2.5)))
            fig.add_vline(x=0.85, line_dash="dot", line_color=GRAY)
            fig.add_hline(y=0.70, line_dash="dot", line_color=GRAY)
            st.plotly_chart(plotly_layout(fig, title="Precision–Recall", xaxis_title="Recall", yaxis_title="Precision",
                                          xaxis_range=[0, 1], yaxis_range=[0, 1.02]), use_container_width=True)
        with c2:
            fig = go.Figure()
            for name, m, color in (("LLM", lm, LLM_COLOR), ("XGBoost", xm, XGB_COLOR)):
                cv = (m or {}).get("curves", {}).get("roc")
                if cv:
                    fig.add_trace(go.Scatter(x=cv["fpr"], y=cv["tpr"], mode="lines", name=name, line=dict(color=color, width=2.5)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GRAY, dash="dot"), showlegend=False))
            st.plotly_chart(plotly_layout(fig, title="ROC", xaxis_title="False positive rate", yaxis_title="True positive rate"),
                            use_container_width=True)

        st.dataframe(df.round(4), use_container_width=True)
        fig = go.Figure()
        for name, color in (("LLM", LLM_COLOR), ("XGBoost", XGB_COLOR)):
            fig.add_trace(go.Bar(name=name, x=names[:2] + names[3:7], y=df[name].iloc[[0, 1, 3, 4, 5, 6]], marker_color=color))
        st.plotly_chart(plotly_layout(fig, title="Side by side", barmode="group", yaxis_range=[0, 1]), use_container_width=True)

        fi = (stats.get("XGBoost") or {}).get("feature_importance") or {}
        if fi:
            top = list(fi.items())[:15][::-1]
            fig = go.Figure(go.Bar(x=[v for _, v in top], y=[k for k, _ in top], orientation="h", marker_color=XGB_COLOR))
            st.plotly_chart(plotly_layout(fig, height=420, title="XGBoost feature importance (gain share)"),
                            use_container_width=True)


# ------------------------------------------------------------------ Live Test
def result_card(name, color, ok, res, ms, server_ms):
    if not ok:
        return f'<div class="card"><h4>{dot(RED)}{name}</h4><div class="muted">{esc(str(res)[:300])}</div></div>'
    p = res["fraud_probability"]
    reasons = "".join(pill(r.split("_", 1)[1].replace("_", " ").lower(), "p-blue") for r in res["reason_codes"]) \
        or '<span class="muted">no reason codes</span>'
    srv = f" · server {server_ms:.0f} ms" if server_ms is not None else ""
    return (f'<div class="card"><h4>{dot(color)}{name} &nbsp;{decision_pill(res["decision"])}</h4>'
            f'<div class="kpi-value">{100 * p:.2f}%</div><div class="kpi-sub">fraud probability · confidence '
            f'{100 * res["confidence"]:.1f}%</div>{prob_bar(p, DECISION_COLOR[res["decision"]])}'
            f'<div style="margin:8px 0">{reasons}</div><div>{esc(res["explanation"])}</div>'
            f'<div class="muted" style="margin-top:8px">round trip {ms:.0f} ms{srv}</div></div>')


with t_live:
    scen = cached_scenarios(xgb_url, xgb_key)
    names = [s["name"] for s in scen]
    top = st.columns([3, 1])
    pick = top[0].selectbox("Scenario", names, index=min(2, len(names) - 1) if len(names) > 2 else 0)
    mode = top[1].radio("Input", ["Form", "Raw prompt"], horizontal=True)
    sc = next(s for s in scen if s["name"] == pick)
    t0 = dict(sc["transaction"])
    if "expect" in sc:
        st.markdown(f'<span class="muted">Scenario expectation:</span> {decision_pill(sc["expect"])}', unsafe_allow_html=True)

    item = None
    if mode == "Form":
        with st.form(f"live_{pick}"):
            st.markdown('<div class="section">Transaction</div>', unsafe_allow_html=True)
            a = st.columns(5)
            amount = a[0].number_input("Amount (₹)", 0.0, 2_000_000.0, float(t0.get("amount", 1000)), step=100.0)
            mcc = a[1].selectbox("MCC", MCC, index=MCC.index(t0.get("mcc", "grocery")) if t0.get("mcc") in MCC else 0)
            channel = a[2].selectbox("Channel", CHANNELS, index=CHANNELS.index(t0.get("channel", "UPI")) if t0.get("channel") in CHANNELS else 0)
            hour = a[3].slider("Hour", 0, 23, int(t0.get("hour", 12)))
            weekday = a[4].selectbox("Weekday", DAYS, index=DAYS.index(t0.get("weekday", "Mon")) if t0.get("weekday") in DAYS else 0)
            st.markdown('<div class="section">Customer</div>', unsafe_allow_html=True)
            b = st.columns(5)
            age = b[0].number_input("Age", 16, 100, int(t0.get("customer_age", 35)))
            acct = b[1].number_input("Account age (days)", 0, 10000, int(t0.get("account_age_days", 365)))
            avg = b[2].number_input("Avg 90d (₹, 0 = unknown)", 0.0, 2_000_000.0, float(t0.get("avg_90d") or 0))
            t30 = b[3].number_input("Txns last 30d", 0, 500, int(t0.get("txns_30d", 0)))
            fails = b[4].number_input("Failed attempts 24h", 0, 20, int(t0.get("fails_24h", 0)))
            st.markdown('<div class="section">Context</div>', unsafe_allow_html=True)
            c = st.columns(5)
            new_dev = c[0].toggle("New device", bool(t0.get("new_device")))
            new_city = c[1].toggle("New city", bool(t0.get("new_city")))
            city = c[2].text_input("Customer city", t0.get("city", "Pune"))
            mcity = c[3].text_input("Merchant city", t0.get("merchant_city", "Pune"))
            dist = c[4].selectbox("Distance (km)", DIST, index=DIST.index(t0.get("dist_km", "<5")) if t0.get("dist_km") in DIST else 0)
            d = st.columns(4)
            ratio = d[0].number_input("Amount ratio vs avg (0 = unknown)", 0.0, 1000.0, float(t0.get("amt_ratio") or 0))
            t1h = d[1].number_input("Txns last hour", 0, 100, int(t0.get("txns_1h", 0)))
            a24 = d[2].number_input("Amount last 24h (₹)", 0.0, 5_000_000.0, float(t0.get("amount_24h", 0)))
            dusers = d[3].number_input("Device users 7d", 0, 20, int(t0.get("device_users_7d", 1)))
            st.markdown('<div class="section">History (most recent first)</div>', unsafe_allow_html=True)
            hist_df = pd.DataFrame(t0.get("history") or [], columns=["days_ago", "channel", "amount", "mcc", "city"])
            hist = st.data_editor(hist_df, num_rows="dynamic", use_container_width=True, key=f"hist_{pick}")
            go_btn = st.form_submit_button("Score with both models")
        if go_btn:
            item = {"transaction_id": t0.get("transaction_id", "LIVE"), "amount": amount, "mcc": mcc, "channel": channel,
                    "hour": hour, "weekday": weekday, "customer_age": age, "account_age_days": acct,
                    "avg_90d": avg or None, "txns_30d": t30, "fails_24h": fails, "new_device": new_dev,
                    "city": city, "merchant_city": mcity, "new_city": new_city, "dist_km": dist,
                    "amt_ratio": ratio or None, "txns_1h": t1h, "amount_24h": a24, "device_users_7d": dusers,
                    "history": [h for h in hist.dropna(subset=["amount"]).to_dict("records")]}
    else:
        raw = st.text_area("fmt-v1 prompt", height=180, value=(
            "[TXN] id=TX_RAW amt=74999.0 mcc=electronics channel=card time=03h Sun\n"
            "[CUST] age=24 acct_age_days=420 avg_90d=1800.00 txns_30d=6 fails_24h=4\n"
            "[CTX] new_device=yes city=Patna merchant_city=Mumbai new_city=yes dist_km=1000+ amt_ratio=41.7 "
            "txns_1h=3 amount_24h=12000.00 device_users_7d=2\n[HIST] none observed\n"
            "[TASK] Assess fraud risk using only the observed context."))
        if st.button("Score prompt with both models"):
            item = {"transaction_id": "TX_RAW", "prompt": raw}

    if item is not None:
        cols = st.columns(2)
        out = {}
        for col, ep, color in ((cols[0], LLM, LLM_COLOR), (cols[1], XGB, XGB_COLOR)):
            ok, res, ms, srv = ep.score_one(item)
            out[ep.name] = (ok, res, ms)
            col.markdown(result_card(ep.name, color, ok, res, ms, srv), unsafe_allow_html=True)
        (lo, lr, lms), (xo, xr, xms) = out["LLM"], out["XGBoost"]
        if lo and xo:
            agree = lr["decision"] == xr["decision"]
            st.markdown(f'<div class="card">{pill("Models agree", "p-allow") if agree else pill("Models disagree", "p-review")}'
                        f' LLM {decision_pill(lr["decision"])} vs XGBoost {decision_pill(xr["decision"])}</div>',
                        unsafe_allow_html=True)
        ss.live_log.append({"time": time.strftime("%H:%M:%S"), "scenario": pick if mode == "Form" else "raw prompt",
                            "llm_decision": lr["decision"] if lo else "error", "llm_p": lr["fraud_probability"] if lo else None,
                            "llm_ms": lms if lo else None, "xgb_decision": xr["decision"] if xo else "error",
                            "xgb_p": xr["fraud_probability"] if xo else None, "xgb_ms": xms if xo else None})

    if ss.live_log:
        st.markdown('<div class="section">Session log</div>', unsafe_allow_html=True)
        log = pd.DataFrame(ss.live_log)
        st.dataframe(log.iloc[::-1], use_container_width=True, hide_index=True)
        k = st.columns(3)
        for i, (col, label) in enumerate((("llm_ms", "LLM"), ("xgb_ms", "XGBoost"))):
            v = log[col].dropna()
            if len(v):
                k[i].markdown(kpi(f"{label} latency p50 / p99", f"{v.quantile(.5):.0f} / {v.quantile(.99):.0f} ms",
                                  f"{len(v)} calls, round trip"), unsafe_allow_html=True)
        if k[2].button("Clear log"):
            ss.live_log = []
            st.rerun()


# ------------------------------------------------------------------ Batch Test
def sample_metrics(y, p, decisions):
    from sklearn.metrics import average_precision_score, roc_auc_score

    y, p = np.asarray(y), np.asarray(p, dtype=float)
    flagged = np.isin(decisions, ["REVIEW", "BLOCK"])
    blocked = np.asarray(decisions) == "BLOCK"
    tp = int((flagged & (y == 1)).sum())
    both = len(np.unique(y)) == 2
    return {"PR-AUC": average_precision_score(y, p) if both else None,
            "ROC-AUC": roc_auc_score(y, p) if both else None,
            "Recall (REVIEW+BLOCK)": tp / max(1, int((y == 1).sum())),
            "Precision (REVIEW+BLOCK)": tp / max(1, int(flagged.sum())),
            "Precision@BLOCK": int((blocked & (y == 1)).sum()) / max(1, int(blocked.sum())),
            "Flagged": int(flagged.sum())}


with t_batch:
    st.markdown('<div class="card"><h4>Live benchmark on labelled samples</h4><div class="muted">Pulls labelled rows '
                'from the XGBoost service (your machine), scores the identical prompts on both models, and computes '
                'metrics on that sample. Or upload your own JSONL / CSV.</div></div>', unsafe_allow_html=True)
    src = st.radio("Source", ["Samples from XGBoost service", "Upload file"], horizontal=True) or "Samples"
    rows = None
    if src.startswith("Samples"):
        c = st.columns(5)
        bsplit = c[0].selectbox("Split", ["holdout", "validation", "train"]) or "holdout"
        n = c[1].number_input("Rows", 10, 2000, 200, step=10)
        fr = c[2].slider("Fraud share", 0.0, 1.0, 0.3, 0.05)
        seed = c[3].number_input("Seed", 0, 10_000, 1)
        c[4].markdown("<br>", unsafe_allow_html=True)
        if c[4].button("Fetch & score", use_container_width=True):
            ok, data, _ = XGB.samples(bsplit, int(n), float(fr), int(seed))
            if not ok:
                st.error(f"Could not fetch samples: {data}")
            else:
                rows = data["rows"]
    else:
        up = st.file_uploader("JSONL (one transaction per line) or CSV; optional column fraud_label", type=["jsonl", "json", "csv"])
        if up is not None and st.button("Score file"):
            txt = up.getvalue().decode("utf-8")
            if up.name.endswith(".csv"):
                rows = pd.read_csv(io.StringIO(txt)).to_dict("records")
            elif txt.strip().startswith("["):
                rows = json.loads(txt)
            else:
                rows = [json.loads(line) for line in txt.splitlines() if line.strip()]
            for i, r in enumerate(rows):
                r.setdefault("transaction_id", f"UP_{i:05d}")
                if isinstance(r.get("history"), str):
                    r["history"] = json.loads(r["history"])

    if rows is not None:
        items = [{k: v for k, v in r.items() if k not in ("fraud_label", "reasons")} for r in rows]
        res = {}
        with st.spinner(f"Scoring {len(items)} rows on both models…"):
            for ep in (LLM, XGB):
                res[ep.name] = ep.score(items)
        df = pd.DataFrame({"transaction_id": [r.get("transaction_id") for r in rows]})
        if all("fraud_label" in r for r in rows):
            df["fraud_label"] = [int(r["fraud_label"]) for r in rows]
        for name, short in (("LLM", "llm"), ("XGBoost", "xgb")):
            ok, data, ms = res[name]
            if ok:
                df[f"{short}_p"] = [d["fraud_probability"] for d in data]
                df[f"{short}_decision"] = [d["decision"] for d in data]
                df[f"{short}_reasons"] = [", ".join(d["reason_codes"]) for d in data]
            else:
                st.error(f"{name}: {data}")
            res[name] = (ok, ms)
        ss.batch = {"df": df, "res": res, "n": len(rows), "prompts": {r.get("transaction_id"): r.get("prompt") for r in rows}}

    if ss.batch is not None:
        df, res, n = ss.batch["df"], ss.batch["res"], ss.batch["n"]
        k = st.columns(4)
        k[0].markdown(kpi("Rows", str(n), f"{int(df['fraud_label'].sum()) if 'fraud_label' in df else '—'} fraud"), unsafe_allow_html=True)
        for col, (name, short) in zip(k[1:3], (("LLM", "llm"), ("XGBoost", "xgb"))):
            ok, ms = res[name]
            col.markdown(kpi(f"{name} time", f"{ms / 1000:.1f} s" if ok else "error",
                             f"{ms / max(1, n):.1f} ms / txn (batched)" if ok else ""), unsafe_allow_html=True)
        if "llm_decision" in df and "xgb_decision" in df:
            agree = (df["llm_decision"] == df["xgb_decision"]).mean()
            k[3].markdown(kpi("Decision agreement", pct(agree), "LLM vs XGBoost"), unsafe_allow_html=True)

        if "fraud_label" in df:
            mt = {}
            for name, short in (("LLM", "llm"), ("XGBoost", "xgb")):
                if f"{short}_p" in df:
                    mt[name] = sample_metrics(df["fraud_label"], df[f"{short}_p"], df[f"{short}_decision"].values)
            if mt:
                st.markdown('<div class="section">Metrics on this sample</div>', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(mt).round(4), use_container_width=True)
                st.caption("Samples are enriched with fraud, so precision here is higher than at the real 0.6 % base rate. "
                           "Use the Benchmark tab (full holdout) for acceptance numbers.")
        if "llm_p" in df and "xgb_p" in df:
            fig = go.Figure()
            color = df["fraud_label"].map({1: RED, 0: GRAY}) if "fraud_label" in df else LLM_COLOR
            fig.add_trace(go.Scatter(x=df["xgb_p"], y=df["llm_p"], mode="markers", text=df["transaction_id"],
                                     marker=dict(color=color, size=8, opacity=.75)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GRAY, dash="dot"), showlegend=False))
            st.plotly_chart(plotly_layout(fig, height=380, title="LLM vs XGBoost probability (red = fraud)",
                                          xaxis_title="XGBoost p", yaxis_title="LLM p", showlegend=False),
                            use_container_width=True)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Download results CSV", df.to_csv(index=False), "sentinel_batch_results.csv", "text/csv")
        with st.expander("Inspect a prompt"):
            tid = st.selectbox("Transaction", df["transaction_id"].tolist())
            p = ss.batch["prompts"].get(tid)
            st.code(p or "(structured input, no prompt stored)", language="text")
