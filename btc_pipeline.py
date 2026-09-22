"""Reproduce the Bitcoin_Price_Forecasting notebook's models and export the
artifacts the Streamlit app reads. Run once; commit data/.

Differences from the notebook, all deliberate:
  * Data runs to today rather than 2025-05-31, so the demo is not stale.
  * Prophet is omitted — it needs a C++ toolchain to build on Windows and would
    also bloat the Streamlit Cloud image. The app treats it as optional.
  * yfinance now returns MultiIndex columns; they are flattened here. The
    notebook's `df['Close']` would break on a fresh run without this.
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
df["Lag1"] = df["Close"].shift(1)
df["Lag7"] = df["Close"].shift(7)
df["Lag30"] = df["Close"].shift(30)
df["Rolling_Mean_7"] = df["Close"].rolling(7).mean()
df["Rolling_Mean_30"] = df["Close"].rolling(30).mean()
df["Rolling_Std_7"] = df["Close"].rolling(7).std()
df = df.dropna()

df["Year"] = df.index.year
df["Month"] = df.index.month
df["Day"] = df.index.day
df["Dayofweek"] = df.index.dayofweek
df["Is_weekend"] = (df.index.dayofweek >= 5).astype(int)

FEATURES = ["Year", "Month", "Day", "Dayofweek", "Is_weekend", "Lag1", "Lag7",
            "Lag30", "Rolling_Mean_7", "Rolling_Mean_30", "Rolling_Std_7"]

train = df.loc[df.index < SPLIT]
test = df.loc[df.index >= SPLIT]
X_train, y_train = train[FEATURES], train["Close"]
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
xgb_pred = best.predict(X_test)
print(f"  best params: {search.best_params_}")


def score(true, pred):
    return {
        "MAE": float(mean_absolute_error(true, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(true, pred))),
        "R2": float(r2_score(true, pred)),
    }


metrics = {"XGBoost": score(y_test, xgb_pred)}
print(f"  XGBoost {metrics['XGBoost']}")

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
    "note": "Prophet omitted; see btc_pipeline.py header.",
}, indent=2))

print("\nwrote:", *[p.name for p in sorted(OUT.iterdir())])
