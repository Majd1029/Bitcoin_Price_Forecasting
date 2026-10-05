"""Reproduce the Bitcoin_Price_Forecasting notebook's models and export the
artifacts the demo page (web/index.html) reads. Run once; commit web/data/.

    python btc_pipeline.py                      # downloads BTC-USD from Yahoo
    BTC_CSV=prices.csv python btc_pipeline.py   # or reads a CSV (date, Close)

Differences from the notebook, all deliberate:
  * Data runs to today rather than 2025-05-31, so the demo is not stale.
  * Prophet is omitted — it needs a C++ toolchain to build on Windows and adds
    a heavy dependency for one extra series.
  * yfinance now returns MultiIndex columns; they are flattened here. The
    notebook's `df['Close']` would break on a fresh run without this.
  * XGBoost predicts LOG-RETURNS, not absolute price.

Why the target changed. The notebook regresses on absolute close. Gradient-
boosted trees predict a piecewise-constant function bounded by the targets they
saw in training, so a model trained on 2018-2022 (BTC peak ~$69k) cannot emit a
value above roughly that, no matter the input. Measured on the 2023-2026 test
window that produced:

    max XGBoost prediction  $63,975
    max actual price       $124,753
    2023 MAE  $1,111   |   2025 MAE  $40,094

Modelling log-returns makes the target stationary, and the price path is
rebuilt as C_hat[t] = C[t-1] * exp(r_hat[t]).

Because that reconstruction is anchored on the previous ACTUAL close, it is a
one-step-ahead forecast and will score a high R2 almost mechanically. A naive
persistence baseline (C_hat[t] = C[t-1]) is therefore computed too: that is the
number XGBoost has to beat for any of this to mean anything. Directional
accuracy is reported for the same reason, with a binomial test against a coin
flip.

Evaluation rules. Every model is scored on the same held-out window (from
SPLIT onward) against the baseline for its horizon, and nothing from that
window is used for fitting, tuning or scaling:
  * XGBoost is tuned with TimeSeriesSplit, so each fold trains on the past
    and validates on what follows. Plain k-fold would let it tune on the future.
  * The LSTM models returns, with scaling fitted on the training years only.
  * 7-day LSTM forecasts are scored by rolling origin: every
    7 days through the test window, forecast the next 7 from data up to that
    day only, against a "price stays flat" baseline.

Volatility. Which way the price moves is close to unpredictable; how much it
moves is not. Volatility clusters (calm weeks follow calm weeks), so next
week's realized volatility is forecast too, with a HAR model (linear in daily,
weekly and monthly realized volatility; Corsi 2009) and XGBoost, against the
baseline "next week is as volatile as last week".
"""
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import binomtest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

OUT = Path(__file__).parent / "web" / "data"
OUT.mkdir(parents=True, exist_ok=True)

TICKER, START, SPLIT = "BTC-USD", "2018-01-01", "2023-01-01"
SEQ_LEN, HORIZON = 60, 7
ANNUALIZE = np.sqrt(365) * 100   # daily log-return std -> annualized %, crypto trades every day
SEED = 42


# ---------------------------------------------------------------- data
def from_yahoo():
    import yfinance as yf

    prices = yf.download(TICKER, start=START, auto_adjust=True, progress=False)
    if isinstance(prices.columns, pd.MultiIndex):   # yfinance returns (field, ticker)
        prices.columns = prices.columns.get_level_values(0)
    return prices[["Close"]] if "Close" in prices else pd.DataFrame(columns=["Close"])


def from_binance():
    """Daily BTC/USDT closes from Binance's public API (no key), 1000 days per
    request. USDT tracks the dollar closely enough for daily closes."""
    import urllib.request

    rows, start_ms = [], int(pd.Timestamp(START).timestamp() * 1000)
    while True:
        url = ("https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d"
               f"&startTime={start_ms}&limit=1000")
        with urllib.request.urlopen(url, timeout=30) as r:
            batch = json.load(r)
        rows += batch
        if len(batch) < 1000:
            break
        start_ms = batch[-1][0] + 1
    # each kline: [open time, open, high, low, close, ...]; drop today's unfinished day
    prices = pd.DataFrame({"Close": [float(k[4]) for k in rows]},
                          index=pd.to_datetime([k[0] for k in rows], unit="ms"))
    return prices[prices.index < pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()]


def load_prices():
    csv = os.getenv("BTC_CSV")
    if csv:
        print(f"reading {csv} ...")
        prices = pd.read_csv(csv, parse_dates=["date"], index_col="date")[["Close"]]
    else:
        prices = pd.DataFrame(columns=["Close"])
        for name, source in (("Yahoo Finance", from_yahoo), ("Binance", from_binance)):
            print(f"downloading BTC-USD from {name} ...")
            try:
                prices = source()
            except Exception as e:
                print(f"  {name} failed: {e}")
                continue
            if len(prices):
                break
            print(f"  {name} returned no data")
    prices = prices.dropna().sort_index()
    if len(prices) < 1000:
        raise SystemExit(
            f"Only {len(prices)} daily prices loaded; need several years. If Yahoo "
            "failed, upgrade yfinance (pip install -U yfinance), or pass a CSV with "
            "date and Close columns: BTC_CSV=prices.csv python btc_pipeline.py")
    return prices


def score(true, pred):
    return {
        "MAE": float(mean_absolute_error(true, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(true, pred))),
        "R2": float(r2_score(true, pred)),
    }


df = load_prices()
print(f"  {len(df)} rows, {df.index.min().date()} -> {df.index.max().date()}")

# ------------------------------------------------------------ features
# target: next-day log return, stationary and unbounded in price space
df["LogRet"] = np.log(df["Close"]).diff()

# features are all lagged returns / return statistics — never absolute price,
# which is what capped the original model
for lag in (1, 2, 3, 7, 14, 30):
    df[f"Ret_Lag{lag}"] = df["LogRet"].shift(lag)
df["Ret_Mean_7"] = df["LogRet"].shift(1).rolling(7).mean()
df["Ret_Mean_30"] = df["LogRet"].shift(1).rolling(30).mean()
df["Ret_Std_7"] = df["LogRet"].shift(1).rolling(7).std()
df["Ret_Std_30"] = df["LogRet"].shift(1).rolling(30).std()
df["PrevClose"] = df["Close"].shift(1)

# volatility features, all known at the close of day t
df["RV_1"] = df["LogRet"].abs() * ANNUALIZE
df["RV_7"] = df["LogRet"].rolling(7).std() * ANNUALIZE
df["RV_30"] = df["LogRet"].rolling(30).std() * ANNUALIZE
# volatility target: realized volatility over the NEXT 7 days (t+1 .. t+7)
df["RV_next7"] = df["LogRet"][::-1].rolling(HORIZON).std()[::-1].shift(-1) * ANNUALIZE

df = df.dropna(subset=[c for c in df.columns if c != "RV_next7"])

df["Month"] = df.index.month
df["Dayofweek"] = df.index.dayofweek
df["Is_weekend"] = (df.index.dayofweek >= 5).astype(int)

FEATURES = ([f"Ret_Lag{l}" for l in (1, 2, 3, 7, 14, 30)]
            + ["Ret_Mean_7", "Ret_Mean_30", "Ret_Std_7", "Ret_Std_30",
               "Month", "Dayofweek", "Is_weekend"])

is_train = df.index < SPLIT
split_pos = int(is_train.sum())          # first test row
train, test = df[is_train], df[~is_train]
print(f"  train {len(train)} / test {len(test)}")
prev = test["PrevClose"].values
actual = test["Close"].values

# ------------------------------------------------------------- xgboost
print("tuning XGBoost (time-series CV) ...")
search = RandomizedSearchCV(
    xgb.XGBRegressor(objective="reg:squarederror", random_state=SEED),
    {
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [2, 3, 4, 6],
        "n_estimators": [50, 100, 200],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    },
    n_iter=15, cv=TimeSeriesSplit(n_splits=5), scoring="neg_mean_absolute_error",
    random_state=SEED, n_jobs=-1, verbose=0,
)
search.fit(train[FEATURES], train["LogRet"])
xgb_ret = search.best_estimator_.predict(test[FEATURES])
xgb_pred = prev * np.exp(xgb_ret)       # rebuild the price path from returns
naive_pred = prev                       # persistence baseline
print(f"  best params: {search.best_params_}")


def direction(pred_prices):
    """Share of days where the predicted move has the right sign, and a one-sided
    binomial test of that share against a coin flip. Flat days are skipped."""
    true_sign, pred_sign = np.sign(actual - prev), np.sign(pred_prices - prev)
    keep = (true_sign != 0) & (pred_sign != 0)
    hits, n = int((true_sign[keep] == pred_sign[keep]).sum()), int(keep.sum())
    return {"DirectionAcc": hits / n,
            "DirectionP": float(binomtest(hits, n, 0.5, alternative="greater").pvalue)}


# ---------------------------------------------------------------- lstm
print("training LSTM on returns ...")
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

tf.keras.utils.set_random_seed(SEED)
ret = df["LogRet"].values
ret_sd = float(train["LogRet"].std())   # scale from the training years only
z = ret / ret_sd

# window ending at position i-1 -> the next HORIZON returns (positions i .. i+6);
# training windows must have every target before the test window starts
starts = np.arange(SEQ_LEN, split_pos - HORIZON + 1)
X_tr = np.stack([z[i - SEQ_LEN:i] for i in starts])[..., None]
y_tr = np.stack([z[i:i + HORIZON] for i in starts])
cut = int(0.9 * len(X_tr))              # chronological validation split
lstm = Sequential([
    Input((SEQ_LEN, 1)),
    LSTM(64, return_sequences=True), Dropout(0.2),
    LSTM(32), Dropout(0.2),
    Dense(HORIZON),
])
lstm.compile(optimizer="adam", loss="mean_squared_error")
lstm.fit(X_tr[:cut], y_tr[:cut], validation_data=(X_tr[cut:], y_tr[cut:]),
         epochs=40, batch_size=32, verbose=0,
         callbacks=[EarlyStopping(monitor="val_loss", patience=5,
                                  restore_best_weights=True)])


def lstm_returns(last_positions):
    """Predicted next-HORIZON log returns from windows ending at each position."""
    X = np.stack([z[p - SEQ_LEN + 1:p + 1] for p in last_positions])[..., None]
    return lstm.predict(X, verbose=0) * ret_sd


# one-day-ahead on the test window: the window ends the day before each test day
lstm_pred = prev * np.exp(lstm_returns(np.arange(split_pos - 1, len(df) - 1))[:, 0])

one_day = {
    "XGBoost": {**score(actual, xgb_pred), **direction(xgb_pred)},
    "LSTM": {**score(actual, lstm_pred), **direction(lstm_pred)},
    "Naive (persistence)": score(actual, naive_pred),
}
for name, m in one_day.items():
    print(f"  1-day {name:20} {m}")

# ----------------------------------------- 7-day forecasts, rolling origin
print("scoring 7-day forecasts by rolling origin ...")
close = df["Close"].values


origins = np.arange(split_pos - 1, len(df) - HORIZON, HORIZON)
truth7 = np.concatenate([close[o + 1:o + 1 + HORIZON] for o in origins])
lstm7 = np.concatenate([close[o] * np.exp(np.cumsum(r))
                        for o, r in zip(origins, lstm_returns(origins))])
naive7 = np.repeat(close[origins], HORIZON)
seven_day = {
    "LSTM": score(truth7, lstm7),
    "Naive (flat)": score(truth7, naive7),
}
for name, m in seven_day.items():
    print(f"  7-day {name:20} {m}")

# ---------------------------------------------------------- volatility
print("forecasting next-week volatility ...")
VOL_FEATURES = ["RV_1", "RV_7", "RV_30"]
# a training row's target covers t+1..t+7, so stop HORIZON days before SPLIT
vol_train = df[df.index < pd.Timestamp(SPLIT) - pd.Timedelta(days=HORIZON)].dropna(subset=["RV_next7"])
vol_test = test.dropna(subset=["RV_next7"])

har = LinearRegression().fit(vol_train[VOL_FEATURES], vol_train["RV_next7"])
har_pred = har.predict(vol_test[VOL_FEATURES])

vol_xgb_features = VOL_FEATURES + ["Ret_Lag1", "Ret_Mean_7", "Dayofweek"]
vol_search = RandomizedSearchCV(
    xgb.XGBRegressor(objective="reg:squarederror", random_state=SEED),
    {"learning_rate": [0.03, 0.1], "max_depth": [2, 3, 4],
     "n_estimators": [100, 200, 400], "subsample": [0.8, 1.0]},
    n_iter=8, cv=TimeSeriesSplit(n_splits=5), scoring="neg_mean_absolute_error",
    random_state=SEED, n_jobs=-1, verbose=0,
)
vol_search.fit(vol_train[vol_xgb_features], vol_train["RV_next7"])
vol_xgb_pred = vol_search.best_estimator_.predict(vol_test[vol_xgb_features])
vol_naive = vol_test["RV_7"].values     # "next week looks like last week"

vol_true = vol_test["RV_next7"].values
volatility = {
    "HAR": score(vol_true, har_pred),
    "XGBoost": score(vol_true, vol_xgb_pred),
    "Naive (last 7 days)": score(vol_true, vol_naive),
}
for name in ("HAR", "XGBoost"):
    volatility[name]["VsNaive"] = 1 - volatility[name]["MAE"] / volatility["Naive (last 7 days)"]["MAE"]
for name, m in volatility.items():
    print(f"  vol   {name:20} {m}")

# -------------------------------------------------------------- export
pd.DataFrame({
    "date": test.index.strftime("%Y-%m-%d"),
    "actual": actual,
    "xgboost": xgb_pred,
    "lstm": lstm_pred,
    "naive": naive_pred,
}).round(2).to_csv(OUT / "predictions.csv", index=False)

last = df.index.max()
future_dates = pd.date_range(last + pd.Timedelta(days=1), periods=HORIZON, freq="D")
pd.DataFrame({
    "date": future_dates.strftime("%Y-%m-%d"),
    "lstm_forecast": close[-1] * np.exp(np.cumsum(lstm_returns([len(df) - 1])[0])),
}).to_csv(OUT / "lstm_forecast.csv", index=False)

pd.DataFrame({
    "date": vol_test.index.strftime("%Y-%m-%d"),
    "actual": vol_true,
    "har": har_pred,
    "xgboost": vol_xgb_pred,
    "naive": vol_naive,
}).round(3).to_csv(OUT / "volatility.csv", index=False)

(OUT / "metrics.json").write_text(json.dumps(
    {"one_day": one_day, "seven_day": seven_day, "volatility": volatility}, indent=2))
(OUT / "meta.json").write_text(json.dumps({
    "ticker": TICKER,
    "data_start": str(df.index.min().date()),
    "data_end": str(last.date()),
    "train_test_split": SPLIT,
    "n_train": int(len(train)),
    "n_test": int(len(test)),
    "n_origins_7day": int(len(origins)),
    "xgb_best_params": {k: (v.item() if hasattr(v, "item") else v)
                        for k, v in search.best_params_.items()},
    "har_coefficients": dict(zip(["intercept"] + VOL_FEATURES,
                                 [float(har.intercept_)] + [float(c) for c in har.coef_])),
    "target": "log-return, price path reconstructed from previous actual close",
    "note": ("All models are scored on the same held-out window against the "
             "baseline for their horizon; tuning uses time-series CV and scaling "
             "uses training years only. Prophet omitted. See btc_pipeline.py header."),
}, indent=2))
print(f"wrote {OUT}")
