"""Streamlit dashboard: live light curve, CME traffic light, rolling MAE vs SILSO.

    streamlit run frontend/dashboard.py
"""
from __future__ import annotations

import os
import time

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

API = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
LIGHTS = {"green": "#2ecc71", "amber": "#f39c12", "red": "#e74c3c"}
LABELS = {"green": "Quiet", "amber": "Elevated", "red": "High CME risk"}

st.set_page_config(page_title="Solar AI Pipeline", page_icon="☀️", layout="wide")


@st.cache_data(ttl=60)
def get_json(path: str, **params):
    try:
        r = httpx.get(f"{API}{path}", params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as exc:
        st.session_state["api_error"] = str(exc)
        return None


def traffic_light(state: str) -> str:
    dots = "".join(
        f'<div style="width:46px;height:46px;border-radius:50%;margin:6px auto;'
        f'background:{LIGHTS[s] if s == state else "#333"};box-shadow:{"0 0 18px " + LIGHTS[s] if s == state else "none"}"></div>'
        for s in ("red", "amber", "green")
    )
    return (
        f'<div style="background:#111;border-radius:14px;padding:8px;width:70px">{dots}</div>'
        f'<p style="margin-top:6px;font-weight:600">{LABELS.get(state, state)}</p>'
    )


with st.sidebar:
    st.header("Settings")
    limit = st.slider("Observations", 24, 5000, 500, step=24)
    window = st.slider("MAE rolling window (days)", 3, 90, 30)
    silso_days = st.slider("SILSO reference span (days)", 30, 730, 180)
    auto = st.checkbox("Auto-refresh (60 s)", value=True)
    if st.button("Refresh now"):
        st.cache_data.clear()
    st.caption(f"API: {API}")

st.title("☀️ Solar AI Pipeline — Global Sunspot Monitor")

obs = get_json("/observations", limit=limit)
metrics = get_json("/metrics") or {}
silso = get_json("/metrics/silso", days=silso_days) or []

if obs is None:
    st.error(f"Cannot reach API at {API}: {st.session_state.get('api_error', '')}")
    st.stop()
if not obs:
    st.info("No observations yet. Drop files into the hot folder or POST to /infer.")
    st.stop()

df = pd.DataFrame(obs)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
latest = df.iloc[-1]

c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
c1.metric("R (corrected) ± σ", f"{latest.r_corrected:.1f} ± {latest.sigma:.1f}")
c2.metric("Groups g / Spots s", f"{int(latest.groups)} / {int(latest.spots)}")
c3.metric("Model confidence", f"{latest.confidence:.0%}")
rate = metrics.get("drdt_per_hour")
c4.metric("dR/dt", f"{rate:+.1f} /h" if rate is not None else "—")
with c5:
    st.markdown(traffic_light(metrics.get("cme_risk", "green")), unsafe_allow_html=True)

st.subheader("Live light curve")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=pd.concat([df.timestamp, df.timestamp[::-1]]),
    y=pd.concat([df.r_corrected + df.sigma, (df.r_corrected - df.sigma)[::-1]]),
    fill="toself", fillcolor="rgba(255,165,0,0.15)", line=dict(width=0), name="±σ", hoverinfo="skip",
))
fig.add_trace(go.Scatter(x=df.timestamp, y=df.r_corrected, mode="lines+markers", name="R corrected", line=dict(color="orange")))
interp = df[df.interpolated]
if not interp.empty:
    fig.add_trace(go.Scatter(x=interp.timestamp, y=interp.r_corrected, mode="markers", name="Spline-interpolated",
                             marker=dict(symbol="x", size=10, color="#9b59b6")))
delta = df[df.delta_class]
if not delta.empty:
    fig.add_trace(go.Scatter(x=delta.timestamp, y=delta.r_corrected, mode="markers", name="δ-class region",
                             marker=dict(symbol="triangle-up", size=12, color="#e74c3c")))
if silso:
    sdf = pd.DataFrame(silso)
    sdf["date"] = pd.to_datetime(sdf["date"], utc=True)
    sdf = sdf[sdf.date >= df.timestamp.min().floor("D")]
    fig.add_trace(go.Scatter(x=sdf.date, y=sdf.sn, mode="lines", name="SILSO daily", line=dict(dash="dot", color="#3498db")))
fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Sunspot number R", legend=dict(orientation="h"))
st.plotly_chart(fig, width="stretch")

st.subheader("Rolling MAE vs SILSO")
if silso:
    ref = pd.DataFrame(silso).assign(date=lambda d: pd.to_datetime(d["date"]).dt.date).set_index("date")["sn"]
    daily = df.assign(date=df.timestamp.dt.date).groupby("date")["r_corrected"].mean()
    joined = pd.concat([daily.rename("pipeline"), ref.rename("silso")], axis=1, join="inner")
    if joined.empty:
        st.info("No overlapping days with SILSO yet (SILSO publishes with a short delay).")
    else:
        joined["abs_err"] = (joined.pipeline - joined.silso).abs()
        joined["rolling_mae"] = joined.abs_err.rolling(window, min_periods=1).mean()
        mfig = go.Figure(go.Scatter(x=joined.index, y=joined.rolling_mae, mode="lines", line=dict(color="#16a085")))
        mfig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), yaxis_title=f"MAE ({window}-day)")
        st.plotly_chart(mfig, width="stretch")
        st.caption(f"Current rolling MAE: {joined.rolling_mae.iloc[-1]:.2f}")
else:
    st.info("SILSO reference unavailable.")

with st.expander("Recent observations"):
    st.dataframe(df.sort_values("timestamp", ascending=False).head(100), width="stretch")

if auto:
    time.sleep(60)
    st.cache_data.clear()
    st.rerun()
