"""Streamlit dashboard: station vs international R comparison, live light curve, CME traffic light, MAE.

    streamlit run frontend/dashboard.py
"""
from __future__ import annotations

import os
import time
from datetime import date, timedelta

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


def send(method: str, path: str, **kw):
    """Write call; clears the read cache and returns (ok, json-or-error-text)."""
    try:
        r = httpx.request(method, f"{API}{path}", timeout=60, **kw)
    except httpx.HTTPError as exc:
        return False, str(exc)
    st.cache_data.clear()
    if r.is_success:
        return True, r.json() if r.content else None
    try:
        return False, r.json().get("detail", r.text)
    except ValueError:
        return False, r.text


def fmt(v, spec=".1f", suffix=""):
    return "—" if v is None else f"{v:{spec}}{suffix}"


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


DRDT_HELP = (
    "dR/dt is how fast the sunspot number R is changing, in R units per hour: "
    "(R_now - R_previous) / hours between them. Example: R going from 50 to 70 in 30 minutes is +40/h. "
    "A large positive value usually means a new active region is emerging; above the alert threshold "
    "it sends a Slack/SMS alert and turns the traffic light amber/red."
)


def comparison_tab():
    countries = get_json("/countries")
    if countries is None:
        st.error(f"Cannot reach API at {API}: {st.session_state.get('api_error', '')}")
        return
    with st.sidebar:
        st.header("Comparison")
        country = st.selectbox("Country", countries) if countries else None
        stations = get_json("/stations", country=country) if country else []
        station = st.selectbox("Station", stations or [], format_func=lambda s: f"{s['name']} ({s.get('city') or '—'})")
        source = st.selectbox(
            "International R source", ["auto", "manual", "silso", "swpc"],
            help="auto = your manually entered values first, then SILSO, then NOAA SWPC.",
        )
        method = st.selectbox("Fill missing days", ["linear", "spline", "none"],
                              help="linear: R_d = R_a + (d-a)/(b-a)·(R_b-R_a) between the nearest observed days.")

    manage_expanders(station)
    if not station:
        st.info("No stations yet. Add one below, or run `python -m backend.seed` to load the TSU Tbilisi sample.")
        return

    full = get_json(f"/stations/{station['id']}/compare", source=source, interpolation=method) or {}
    rows = full.get("rows", [])
    if not rows:
        st.info(f"{station['name']} has no daily counts yet. Add some in “Add daily count” or import a CSV below.")
        return
    first, last = date.fromisoformat(rows[0]["date"]), date.fromisoformat(rows[-1]["date"])

    c1, c2 = st.columns([1, 2])
    day = c1.date_input("Date", value=last, help="Pick any day; days without an observation are interpolated.")
    rng = c2.date_input("Date range for chart & statistics", value=(first, last))
    start, end = (rng if isinstance(rng, tuple) and len(rng) == 2 else (first, last))

    sel = get_json(f"/stations/{station['id']}/compare", source=source, interpolation=method,
                   start=day.isoformat(), end=day.isoformat()) or {}
    data = get_json(f"/stations/{station['id']}/compare", source=source, interpolation=method,
                    start=start.isoformat(), end=end.isoformat()) or {}
    summ = full.get("summary", {})
    k_fit = summ.get("k_fit")

    st.subheader(f"{station['name']}, {station['country']} — {day:%d %B %Y}")
    row = (sel.get("rows") or [None])[0]
    if row is None or row.get("r_local") is None:
        ref_val = row.get("r_international") if row else None
        st.warning(
            f"No local count or interpolation for this day (station data covers {first} → {last})."
            + (f" International R = {ref_val:.0f}." if ref_val is not None else "")
        )
    else:
        kind = "observed" if row["observed"] else f"{method}-interpolated"
        m = st.columns(5)
        m[0].metric("Raw W = 10g + s", fmt(row["wolf_raw"], ".0f"),
                    help=f"g={row['groups']}, s={row['spots']}" if row["observed"] else "interpolated")
        m[1].metric(f"Local R (k={fmt(row['k'], '.2f')})" if row["observed"] else "Local R", fmt(row["r_local"]),
                    delta=fmt(row["pct_diff_local"], "+.1f", "% vs intl") if row["pct_diff_local"] is not None else None,
                    delta_color="off")
        m[2].metric(f"Calibrated R (k_fit={fmt(k_fit, '.3f')})", fmt(row["r_calibrated"]),
                    delta=fmt(row["pct_diff_calibrated"], "+.1f", "% vs intl") if row["pct_diff_calibrated"] is not None else None,
                    delta_color="off")
        m[3].metric("International R", fmt(row["r_international"], ".0f"),
                    help=f"source: {row['reference_source']}" if row["reference_source"] else "no reference value for this day")
        has_both = row["r_international"] is not None and row["r_calibrated"] is not None
        diff = row["r_calibrated"] - row["r_international"] if has_both else None
        m[4].metric("Calibrated − International", fmt(diff, "+.1f"))
        st.caption(
            f"This day is **{kind}**. International source: **{row['reference_source'] or 'none'}**. "
            "% difference = (R − R_intl) / R_intl × 100. "
            "Calibrated R uses k_fit = Σ(W·R_intl)/Σ(W²), fitted by least squares on every observed day "
            f"that has an international value (n = {summ.get('n_fit', 0)}"
            + (f", σ_k = {summ['k_fit_sigma']:.3f}" if summ.get("k_fit_sigma") else "") + ")."
        )
        for note in summ.get("notes", []):
            st.caption(f"⚠️ {note}")

    df = pd.DataFrame(data.get("rows", []))
    if df.empty:
        return
    df["date"] = pd.to_datetime(df["date"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.date, y=df.r_international, mode="lines+markers", name="International R",
                             line=dict(color="#3498db"), connectgaps=False))
    fig.add_trace(go.Scatter(x=df.date, y=df.r_local, mode="lines", name="Local R (k)", line=dict(color="orange")))
    fig.add_trace(go.Scatter(x=df.date, y=df.r_calibrated, mode="lines", name="Calibrated R (k_fit)",
                             line=dict(color="#2ecc71", dash="dash")))
    obs_df = df[df.observed]
    fig.add_trace(go.Scatter(x=obs_df.date, y=obs_df.r_local, mode="markers", name="Observed day",
                             marker=dict(size=9, color="orange", line=dict(width=1, color="black"))))
    fig.add_vline(x=pd.Timestamp(day), line=dict(color="grey", dash="dot"))
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Sunspot number R",
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")

    s = data.get("summary", {})
    st.markdown("**Statistics over the selected range (observed days only)**")
    a = st.columns(4)
    a[0].metric("Days observed / filled", f"{s.get('n_observed', 0)} / {s.get('n_interpolated', 0)}")
    a[1].metric("Mean local R", fmt(s.get("mean_r_local"), ".2f"))
    a[2].metric("σ local R", fmt(s.get("std_r_local"), ".2f"), help="Population standard deviation")
    a[3].metric("Mean international R", fmt(s.get("mean_r_international"), ".2f"))
    b = st.columns(4)
    b[0].metric("MAE local", fmt(s.get("mae_local"), ".2f"), help="Mean absolute error of local R vs international R")
    b[1].metric("MAE calibrated", fmt(s.get("mae_calibrated"), ".2f"), help="Mean absolute error of calibrated R vs international R")
    b[2].metric("Mean % diff local", fmt(s.get("mean_pct_diff_local"), "+.1f", "%"))
    b[3].metric("Mean % diff calibrated", fmt(s.get("mean_pct_diff_calibrated"), "+.1f", "%"))

    show = df.assign(date=df.date.dt.date).rename(columns={
        "groups": "g", "spots": "s", "wolf_raw": "W", "r_local": "R local", "r_calibrated": "R calibrated",
        "r_international": "R intl", "reference_source": "intl source", "pct_diff_local": "% diff local",
        "pct_diff_calibrated": "% diff calibrated"})
    with st.expander("Daily table", expanded=False):
        st.dataframe(show.round(2), width="stretch", hide_index=True)
        st.download_button("Download CSV", show.to_csv(index=False), f"{station['name']}_comparison.csv", "text/csv")


def manage_expanders(station):
    if station:
        with st.expander("Add / edit a daily count"):
            with st.form("count"):
                c = st.columns(5)
                d = c[0].date_input("Date", value=date.today(), key="count_date")
                g = c[1].number_input("Groups g", 0, 200, 0)
                s = c[2].number_input("Spots s", 0, 2000, 0)
                k = c[3].number_input("k (blank = station default)", 0.0, 10.0, float(station["k_factor"]), 0.05)
                ri = c[4].number_input("International R (optional, 0 = skip)", 0.0, 1000.0, 0.0)
                notes = st.text_input("Notes")
                if st.form_submit_button("Save count"):
                    body = {"date": d.isoformat(), "groups": int(g), "spots": int(s), "k": k or None,
                            "notes": notes or None, "international_r": ri or None}
                    ok, res = send("POST", f"/stations/{station['id']}/counts", json=body)
                    (st.success if ok else st.error)("Saved." if ok else res)
        with st.expander("Import CSV"):
            st.caption("Columns: `date,g,s` plus optional `k,ri,seeing,notes`. Dates as YYYY-MM-DD or DD.MM.YY. "
                       "`ri` is stored as a manual international value.")
            up = st.file_uploader("CSV file", type="csv")
            if up is not None and st.button("Import"):
                ok, res = send("POST", f"/stations/{station['id']}/counts/import",
                               files={"file": (up.name, up.getvalue(), "text/csv")})
                (st.success if ok else st.error)(f"Imported {res['imported']} rows." if ok else res)
    with st.expander("International R: manual values & refresh"):
        c = st.columns([1, 1, 1, 2])
        d = c[0].date_input("Date", value=date.today() - timedelta(days=1), key="ref_date")
        v = c[1].number_input("R_intl", 0.0, 1000.0, 0.0, key="ref_val")
        if c[2].button("Save manual value"):
            ok, res = send("PUT", f"/reference/{d.isoformat()}", json={"value": v})
            (st.success if ok else st.error)("Saved." if ok else res)
        if c[3].button("Refresh SILSO + NOAA now"):
            ok, res = send("POST", "/reference/refresh")
            (st.info if ok else st.error)(res)
        st.caption("Manual values override SILSO, which overrides NOAA SWPC. Remote values are cached in the database.")
    with st.expander("Add a station / country"):
        with st.form("station"):
            c = st.columns(4)
            name = c[0].text_input("Station name")
            country = c[1].text_input("Country")
            city = c[2].text_input("City")
            kf = c[3].number_input("Default k", 0.05, 10.0, 1.0, 0.05)
            if st.form_submit_button("Create station"):
                ok, res = send("POST", "/stations", json={"name": name, "country": country, "city": city or None, "k_factor": kf})
                (st.success if ok else st.error)("Created — pick it in the sidebar." if ok else res)


def live_tab():
    with st.sidebar:
        st.header("Live pipeline")
        limit = st.slider("Observations", 24, 5000, 500, step=24)
        window = st.slider("MAE rolling window (days)", 3, 90, 30)
        silso_days = st.slider("International reference span (days)", 30, 730, 180)
        auto = st.checkbox("Auto-refresh live tab (60 s)", value=False)
        if st.button("Refresh now"):
            st.cache_data.clear()
        st.caption(f"API: {API}")
    st.session_state['auto_refresh'] = auto

    obs = get_json("/observations", limit=limit)
    metrics = get_json("/metrics") or {}
    silso = get_json("/metrics/silso", days=silso_days) or []

    if obs is None:
        st.error(f"Cannot reach API at {API}: {st.session_state.get('api_error', '')}")
        return
    if not obs:
        st.info("No observations yet. Drop files into the hot folder or POST to /infer.")
        return

    df = pd.DataFrame(obs)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    latest = df.iloc[-1]

    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
    c1.metric("R (corrected) ± σ", f"{latest.r_corrected:.1f} ± {latest.sigma:.1f}")
    c2.metric("Groups g / Spots s", f"{int(latest.groups)} / {int(latest.spots)}")
    c3.metric("Model confidence", f"{latest.confidence:.0%}")
    rate = metrics.get("drdt_per_hour")
    c4.metric("dR/dt (change in R per hour)", f"{rate:+.1f} /h" if rate is not None else "—", help=DRDT_HELP)
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
        fig.add_trace(go.Scatter(x=sdf.date, y=sdf.sn, mode="lines", name="International daily R", line=dict(dash="dot", color="#3498db")))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Sunspot number R", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Rolling MAE vs international R")
    if silso:
        ref = pd.DataFrame(silso).assign(date=lambda d: pd.to_datetime(d["date"]).dt.date).set_index("date")["sn"]
        daily = df.assign(date=df.timestamp.dt.date).groupby("date")["r_corrected"].mean()
        joined = pd.concat([daily.rename("pipeline"), ref.rename("silso")], axis=1, join="inner")
        if joined.empty:
            st.info("No overlapping days with the international reference yet (it publishes with a short delay).")
        else:
            joined["abs_err"] = (joined.pipeline - joined.silso).abs()
            joined["rolling_mae"] = joined.abs_err.rolling(window, min_periods=1).mean()
            mfig = go.Figure(go.Scatter(x=joined.index, y=joined.rolling_mae, mode="lines", line=dict(color="#16a085")))
            mfig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), yaxis_title=f"MAE ({window}-day)")
            st.plotly_chart(mfig, width="stretch")
            st.caption(f"Current rolling MAE: {joined.rolling_mae.iloc[-1]:.2f}")
    else:
        st.info("International reference unavailable (SILSO and NOAA could not be reached and nothing is cached).")

    with st.expander("Recent observations"):
        st.dataframe(df.sort_values("timestamp", ascending=False).head(100), width="stretch")

    with st.expander("What is dR/dt?"):
        st.markdown(DRDT_HELP)


st.title("☀️ Solar AI Pipeline — Global Sunspot Monitor")
tab_cmp, tab_live = st.tabs(["Country vs international R", "Live image pipeline"])
with tab_cmp:
    comparison_tab()
with tab_live:
    live_tab()

if st.session_state.get('auto_refresh'):
    time.sleep(60)
    st.cache_data.clear()
    st.rerun()
