# Bitcoin Price Forecasting

**[Try the live demo](https://bitcoinpriceforecasting-web.vercel.app)** — interactive actual-vs-predicted chart with the naive baseline shown alongside each model.

XGBoost and an LSTM on BTC-USD daily closes (2018 - today),
plus a forecast of next week's volatility.

- **Live demo:** a static page on Vercel (`web/index.html`), with a live BTC price from CoinGecko
- **Notebook:** original exploration and model development
- **`btc_pipeline.py`:** trains offline and exports `web/data/`; the page only renders

## How the models are evaluated

Test window: 2023-01-01 onward. Nothing from it is used for fitting, tuning or
scaling, and every model is compared with the naive baseline for its horizon.

| Question | Models | Baseline | Scored by |
|---|---|---|---|
| Tomorrow's price | XGBoost, LSTM (both on log-returns) | tomorrow = today | MAE, RMSE, R², directional accuracy with a binomial test vs. a coin flip |
| Price over the next 7 days | LSTM | price stays flat | rolling origin: every 7 days, forecast the next 7 using only data up to that day |
| Next week's volatility | HAR, XGBoost | next week is as volatile as last week | MAE, RMSE, R² on annualized realized volatility |

Details that matter:

- XGBoost is tuned with `TimeSeriesSplit`, so each fold trains on the past and
  validates on what follows. Plain k-fold cross-validation would tune on the future.
- The LSTM's scaling is fitted on the training years only.
- HAR (Corsi, 2009) is a linear model of the last day's, week's and month's
  volatility: simple, and a standard benchmark in volatility forecasting.

The latest numbers are on the live demo and in `web/data/metrics.json`.

## Results, and an honest reading of them

**On price, no model beats the naive baseline.** Predicting each day's close as
the previous day's close is as accurate as XGBoost or the LSTM, and their
directional accuracy is statistically indistinguishable from a coin flip.

The R² near 0.997 looks impressive and means almost nothing. Each prediction is
anchored on the previous *actual* close, so nearly all the explained variance is
yesterday's price rather than model skill. Any one-step-ahead price model scores
like this, which is exactly why the persistence baseline is reported next to it.

This is the expected result. Daily crypto returns carry little signal
recoverable from lagged returns, and a model that says otherwise usually has a
leak.

**Volatility is a different story.** Volatility clusters, with calm weeks
following calm weeks, so how much the price will move is far more predictable
than which way. The HAR model forecasts next week's volatility clearly better
than the naive baseline. That contrast, not a price prediction, is the finding.

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
  the pipeline for one extra series.
- Not financial advice. A modelling exercise on historical data.

## Run locally

The page is static, so any web server works:

```bash
cd web
python -m http.server 8000      # then open http://localhost:8000
```

To retrain the models and refresh the data:

```bash
pip install -r requirements.txt
python btc_pipeline.py          # rewrites web/data/ (Yahoo Finance, or Binance if Yahoo fails)
BTC_CSV=prices.csv python btc_pipeline.py   # or from a CSV with date and Close columns
```

## Deploying

On vercel.com, **Add New → Project**, import this repository, set **Root
Directory** to `web`, Framework Preset **Other**, no build command. Every push
to `main` redeploys.
