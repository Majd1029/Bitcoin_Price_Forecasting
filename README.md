# Bitcoin Price Forecasting

XGBoost, an LSTM and a KMeans trend model on BTC-USD daily closes (2018 - today).

- **Live demo:** Streamlit Community Cloud, deployed from `app.py`
- **Notebook:** original exploration and model development
- **`btc_pipeline.py`:** trains offline and exports `data/`; the app only renders

## Results, and an honest reading of them

Test window 2023-01-01 onward, 1361 days.

| Model | MAE | RMSE | R² | Directional accuracy |
|---|---|---|---|---|
| XGBoost (log-returns) | $1,133 | $1,699 | 0.9966 | **49.9%** |
| **Naive persistence** | **$1,130** | **$1,697** | **0.9966** | — |
| LSTM (7-day, scaled) | $2,852 | $4,040 | 0.8465 | — |
| KMeans + linear trend | $5,391 | $6,547 | -1.974 | — |

**XGBoost does not beat the naive baseline.** Predicting each day's close as the
previous day's close scores marginally better. Directional accuracy is 49.9% —
a coin flip.

The R² of 0.9966 looks impressive and means almost nothing. Each prediction is
anchored on the previous *actual* close, so nearly all the explained variance is
yesterday's price rather than model skill. Any one-step-ahead price model scores
like this, which is exactly why the persistence baseline is reported next to it.

This is the expected result. Daily crypto returns carry little signal
recoverable from lagged returns, and a model that says otherwise usually has a
leak. The finding is the deliverable.

## The extrapolation bug this replaced

The first version regressed on absolute close price, as the notebook does.
Gradient-boosted trees emit a piecewise-constant function bounded by the targets
seen in training, so a model trained on 2018-2022 (BTC peak ~$69k) cannot
predict above roughly that value regardless of input:

| | |
|---|---|
| max XGBoost prediction | $63,975 |
| max actual price in test | $124,753 |

Error by year made it obvious:

| Year | MAE | Actual avg | Predicted avg |
|---|---|---|---|
| 2023 | $1,111 | $28,859 | $27,987 |
| 2024 | $7,805 | $65,964 | $58,404 |
| 2025 | $40,094 | $101,642 | $61,548 |

Switching the target to log-returns and rebuilding the price path as
`C[t-1] * exp(r_hat[t])` removes the ceiling — max prediction is now $124,974.
It did not, however, make the model useful, per the table above.

## Notes

- `yfinance` now returns MultiIndex columns; the pipeline flattens them. The
  notebook's `df['Close']` breaks on a fresh run without this.
- Prophet is omitted: it needs a C++ toolchain at install time and would bloat
  the deployment for one extra series.
- Not financial advice. A modelling exercise on historical data.

## Run locally

```bash
pip install -r requirements.txt
python btc_pipeline.py     # regenerate data/ (optional)
streamlit run app.py
```
