import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).parent / "data"

st.set_page_config(page_title="Bitcoin Price Forecasting", page_icon="₿", layout="wide")
st.title("₿ Bitcoin Price Forecasting")
st.caption(
    "XGBoost, an LSTM and a KMeans trend model on BTC-USD daily closes. "
    "Models are trained offline; this app renders the stored predictions."
)


@st.cache_data
def load():
    preds = pd.read_csv(DATA / "predictions.csv", parse_dates=["date"])
    metrics = json.loads((DATA / "metrics.json").read_text())
    lstm_path = DATA / "lstm_forecast.csv"
    lstm = pd.read_csv(lstm_path, parse_dates=["date"]) if lstm_path.exists() else None
    meta_path = DATA / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return preds, metrics, lstm, meta


@st.cache_data(ttl=900, show_spinner=False)
def live_price():
    """Current BTC price, from whichever public source answers first.

    yfinance alone is not reliable here: Yahoo throttles or blocks shared
    datacenter IPs, so on a hosted runner it returns empty frames or raises
    while working fine from a laptop. CoinGecko and Coinbase both permit
    server-side calls without a key.

    Returns (price, change_usd, source) or (None, None, reason).
    """
    import json
    import urllib.request

    def _get(url, timeout=6):
        req = urllib.request.Request(url, headers={"User-Agent": "portfolio-demo"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)

    # CoinGecko: price plus 24h change in one call
    try:
        d = _get("https://api.coingecko.com/api/v3/simple/price"
                 "?ids=bitcoin&vs_currencies=usd&include_24h_change=true")
        px = float(d["bitcoin"]["usd"])
        pct = d["bitcoin"].get("usd_24h_change")
        chg = px * float(pct) / 100.0 if pct is not None else None
        return px, chg, "CoinGecko"
    except Exception:
        pass

    # Coinbase: spot only, no change
    try:
        d = _get("https://api.coinbase.com/v2/prices/BTC-USD/spot")
        return float(d["data"]["amount"]), None, "Coinbase"
    except Exception:
        pass

    # yfinance last, since it is the one that fails on hosted runners
    try:
        import yfinance as yf

        hist = yf.Ticker("BTC-USD").history(period="2d")
        if len(hist) >= 2:
            latest, prev = hist["Close"].iloc[-1], hist["Close"].iloc[-2]
            return float(latest), float(latest - prev), "Yahoo"
    except Exception:
        pass

    return None, None, "no price source reachable"


preds, metrics, lstm, meta = load()

# ---- metric row ---------------------------------------------------------
cols = st.columns(len(metrics) + 1)
price, delta, source = live_price()
cols[0].metric(
    "BTC-USD now",
    f"${price:,.0f}" if price else "unavailable",
    f"{delta:+,.0f} (24h)" if delta else None,
    help=(f"Live spot from {source}, cached 15 min."
          if price else
          f"Live price unavailable: {source}. The forecasts below are "
          "precomputed and unaffected."),
)
for col, (name, m) in zip(cols[1:], metrics.items()):
    extra = (f"dir. {m['DirectionAcc']:.1%}" if "DirectionAcc" in m
             else f"MAE ${m['MAE']:,.0f}")
    col.metric(f"{name} · R²", f"{m['R2']:.3f}", extra, delta_color="off")

xgb_m, naive_m = metrics.get("XGBoost"), metrics.get("Naive (persistence)")
if xgb_m and naive_m:
    beats = naive_m["MAE"] - xgb_m["MAE"]
    st.warning(
        f"**XGBoost does not beat a naive baseline.** Predicting each day's "
        f"close as the previous day's close gives MAE ${naive_m['MAE']:,.0f}; "
        f"XGBoost gives ${xgb_m['MAE']:,.0f} — a difference of ${beats:,.0f}, "
        f"in the noise. Its directional accuracy is "
        f"{xgb_m['DirectionAcc']:.1%}, i.e. a coin flip.\n\n"
        "The R² near 0.997 is an artifact: each prediction is anchored on the "
        "previous *actual* close, so almost all the explained variance is "
        "yesterday's price, not skill. This is the expected result — daily "
        "crypto returns carry little signal recoverable from lagged returns."
    )

# ---- controls -----------------------------------------------------------
available = [c for c in ("xgboost", "naive", "prophet") if c in preds.columns]
left, right = st.columns([3, 1])
with right:
    LABELS = {"xgboost": "XGBoost", "naive": "Naive (persistence)",
              "prophet": "Prophet"}
    shown = st.multiselect("Models", available, default=available,
                           format_func=lambda c: LABELS.get(c, c.capitalize()))
    # Prophet is not part of this pipeline (see btc_pipeline.py), so the
    # control only appears if prophet columns are actually present.
    band = (st.checkbox("Prophet confidence band", value=True)
            if "prophet_lo" in preds.columns else False)
with left:
    lo, hi = preds["date"].min().date(), preds["date"].max().date()
    start, end = st.slider("Test window", lo, hi, (lo, hi), format="YYYY-MM-DD")

view = preds[(preds["date"].dt.date >= start) & (preds["date"].dt.date <= end)]

# ---- chart --------------------------------------------------------------
COLORS = {"xgboost": "#f7931a", "naive": "#888888", "prophet": "#4c9be8"}
fig = go.Figure()

if band and "prophet_lo" in view.columns and "prophet" in shown:
    fig.add_trace(go.Scatter(x=view["date"], y=view["prophet_hi"], line_width=0,
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=view["date"], y=view["prophet_lo"], line_width=0,
                             fill="tonexty", fillcolor="rgba(76,155,232,0.15)",
                             name="Prophet 80% interval", hoverinfo="skip"))

fig.add_trace(go.Scatter(x=view["date"], y=view["actual"], name="Actual",
                         line=dict(color="#e8e8e8", width=2)))
for m in shown:
    fig.add_trace(go.Scatter(x=view["date"], y=view[m],
                             name=LABELS.get(m, m.capitalize()),
                             line=dict(color=COLORS.get(m), width=2, dash="dot")))

if lstm is not None:
    FORECASTS = {
        "lstm_forecast": ("LSTM 7-day horizon", "#2ecc71"),
        "kmeans_forecast": ("KMeans+LR 7-day horizon", "#e056a0"),
    }
    for col, (label, colour) in FORECASTS.items():
        if col in lstm.columns:
            fig.add_trace(go.Scatter(x=lstm["date"], y=lstm[col], name=label,
                                     mode="lines+markers",
                                     line=dict(color=colour, width=2, dash="dash")))

fig.update_layout(
    height=520, hovermode="x unified",
    margin=dict(l=0, r=0, t=10, b=0),
    yaxis_title="Price (USD)", xaxis_title=None,
    legend=dict(orientation="h", y=1.08, x=0),
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
)
fig.update_xaxes(showgrid=False)
fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)", tickprefix="$")
st.plotly_chart(fig, use_container_width=True)

# ---- residuals ----------------------------------------------------------
with st.expander("Residuals (actual − predicted)"):
    resid = pd.DataFrame({"date": view["date"]})
    for m in shown:
        resid[LABELS.get(m, m.capitalize())] = view["actual"] - view[m]
    st.line_chart(resid.set_index("date"))

if meta:
    st.caption(
        f"{meta.get('ticker','BTC-USD')} · {meta.get('data_start')} to "
        f"{meta.get('data_end')} · train/test split {meta.get('train_test_split')} "
        f"({meta.get('n_train')} train / {meta.get('n_test')} test rows). "
        f"{meta.get('note','')}"
    )

st.markdown(
    "Predictions are precomputed — the models are trained offline and this app "
    "only renders the results. **Not financial advice**; a modelling exercise on "
    "historical data, and past fit says little about future price. "
    "[Notebook & source](https://github.com/Majd1029/Bitcoin_Price_Forecasting)"
)
