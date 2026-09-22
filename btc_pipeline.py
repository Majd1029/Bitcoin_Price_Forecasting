"""Reproduce the Bitcoin_Price_Forecasting notebook's models and export the
artifacts the Streamlit app reads. Run once; commit data/.

Differences from the notebook, all deliberate:
  * Data runs to today rather than 2025-05-31, so the demo is not stale.
  * Prophet is omitted — it needs a C++ toolchain to build on Windows and would
    also bloat the Streamlit Cloud image. The app treats it as optional.
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
accuracy is reported for the same reason.
"""
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler

OUT = Path("btc_data")
OUT.mkdir(exist_ok=True)

TICKER, START, SPLIT = "BTC-USD", "2018-01-01", "2023-01-01"
SEQ_LEN, HORIZON = 60, 7

# ---------------------------------------------------------------- data
print("downloading BTC-USD ...")
df = yf.download(TICKER, start=START, auto_adjust=True, progress=False)
if isinstance(df.columns, pd.MultiIndex):           # yfinance >= 0.2.51
    df.columns = df.columns.get_level_values(0)
df = df[["Close"]].dropna()
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
df = df.dropna()

df["Year"] = df.index.year
df["Month"] = df.index.month
df["Day"] = df.index.day
df["Dayofweek"] = df.index.dayofweek
df["Is_weekend"] = (df.index.dayofweek >= 5).astype(int)

FEATURES = ([f"Ret_Lag{l}" for l in (1, 2, 3, 7, 14, 30)]
            + ["Ret_Mean_7", "Ret_Mean_30", "Ret_Std_7", "Ret_Std_30",
               "Month", "Dayofweek", "Is_weekend"])

train = df.loc[df.index < SPLIT]
test = df.loc[df.index >= SPLIT]
X_train, y_train = train[FEATURES], train["LogRet"]
X_test, y_test = test[FEATURES], test["Close"]
print(f"  train {len(train)} / test {len(test)}")

# ------------------------------------------------------------- xgboost
print("tuning XGBoost ...")
search = RandomizedSearchCV(
    xgb.XGBRegressor(objective="reg:squarederror"),
    {
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [4, 6, 8],
        "n_estimators": [50, 100, 150],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    },
    n_iter=10, cv=5, random_state=42, n_jobs=-1, verbose=0,
)
search.fit(X_train, y_train)
best = search.best_estimator_
xgb_ret = best.predict(X_test)
# rebuild the price path from predicted returns
xgb_pred = test["PrevClose"].values * np.exp(xgb_ret)
naive_pred = test["PrevClose"].values          # persistence baseline
print(f"  best params: {search.best_params_}")


def score(true, pred):
    return {
        "MAE": float(mean_absolute_error(true, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(true, pred))),
        "R2": float(r2_score(true, pred)),
    }


def direction_acc(true_prices, pred_prices, prev):
    return float((np.sign(np.asarray(true_prices) - prev)
                  == np.sign(np.asarray(pred_prices) - prev)).mean())

prev = test["PrevClose"].values
metrics = {
    "XGBoost": score(y_test, xgb_pred),
    "Naive (persistence)": score(y_test, naive_pred),
}
# Directional accuracy is only defined for a model that predicts a change.
# Persistence predicts none, so it gets no DirectionAcc key at all rather than
# a misleading 0%.
metrics["XGBoost"]["DirectionAcc"] = direction_acc(y_test, xgb_pred, prev)
print(f"  XGBoost {metrics['XGBoost']}")
print(f"  Naive   {metrics['Naive (persistence)']}")
print(f"  max xgb prediction ${xgb_pred.max():,.0f} vs max actual ${y_test.max():,.0f}")

# ------------------------------------------------------ kmeans + trend
scaler_k = MinMaxScaler()
scaled = scaler_k.fit_transform(df[["Close"]])
df["Cluster"] = KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(scaled)
recent = df[df["Cluster"] == df["Cluster"].iloc[-1]]["Close"].iloc[-30:]
coeffs = np.polyfit(np.arange(len(recent)), recent.values, 1)
kmeans_future = np.polyval(coeffs, np.arange(len(recent), len(recent) + HORIZON))
metrics["KMeans+LR"] = score(df["Close"].iloc[-HORIZON:].values, kmeans_future)
print(f"  KMeans+LR {metrics['KMeans+LR']}")

# ---------------------------------------------------------------- lstm
print("training LSTM ...")
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

scaler_l = MinMaxScaler()
series = scaler_l.fit_transform(df[["Close"]])

Xs, ys = [], []
for i in range(SEQ_LEN, len(series) - HORIZON):
    Xs.append(series[i - SEQ_LEN:i, 0])
    ys.append(series[i:i + HORIZON, 0])
Xs = np.array(Xs).reshape(-1, SEQ_LEN, 1)
ys = np.array(ys)

cut = int(0.9 * len(Xs))
model = Sequential([
    Input((SEQ_LEN, 1)),
    LSTM(100, return_sequences=True), Dropout(0.2),
    LSTM(100), Dropout(0.2),
    Dense(HORIZON),
])
model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(
    Xs[:cut], ys[:cut], validation_data=(Xs[cut:], ys[cut:]),
    epochs=40, batch_size=32, verbose=0,
    callbacks=[EarlyStopping(monitor="val_loss", patience=5,
                             restore_best_weights=True)],
)

lstm_test = model.predict(Xs[cut:], verbose=0)
metrics["LSTM"] = score(
    scaler_l.inverse_transform(ys[cut:]).ravel(),
    scaler_l.inverse_transform(lstm_test).ravel(),
)
print(f"  LSTM {metrics['LSTM']}")

lstm_future = scaler_l.inverse_transform(model.predict(Xs[-1:], verbose=0)).ravel()

# -------------------------------------------------------------- export
pd.DataFrame({
    "date": test.index,
    "actual": np.asarray(y_test).ravel(),
    "xgboost": np.asarray(xgb_pred).ravel(),
    "naive": np.asarray(naive_pred).ravel(),
}).to_csv(OUT / "predictions.csv", index=False)

last = df.index.max()
pd.DataFrame({
    "date": pd.date_range(last + pd.Timedelta(days=1), periods=HORIZON, freq="D"),
    "lstm_forecast": lstm_future,
    "kmeans_forecast": kmeans_future,
}).to_csv(OUT / "lstm_forecast.csv", index=False)

(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
(OUT / "meta.json").write_text(json.dumps({
    "ticker": TICKER,
    "data_start": str(df.index.min().date()),
    "data_end": str(df.index.max().date()),
    "train_test_split": SPLIT,
    "n_train": int(len(train)),
    "n_test": int(len(test)),
    "xgb_best_params": {k: (v.item() if hasattr(v, "item") else v)
                        for k, v in search.best_params_.items()},
    "target": "log-return, price path reconstructed from previous actual close",
    "note": ("XGBoost predicts log-returns; a naive persistence baseline is "
             "included because one-step-ahead price R2 is high by construction. "
             "Prophet omitted. See btc_pipeline.py header."),
}, indent=2))

print("\nwrote:", *[p.name for p in sorted(OUT.iterdir())])
