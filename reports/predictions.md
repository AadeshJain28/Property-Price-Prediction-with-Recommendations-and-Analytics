# Predictions stated before the run, and what happened

| # | Prediction | Held? | Measurement |
|---|---|---|---|
| P1 | Grouped split scores materially worse than the shuffled split | NO | best R2(log): grouped 0.8373 vs shuffled 0.8159, gap -0.0214 |
| P2 | R2 on price is lower than R2 on log1p(price) | YES | tuned test: r2_price 0.7196 vs r2_log 0.8265 |
| P3 | XGBoost tops the leaderboard | YES | winner: xgboost; XGB - RF margin on R2(log) = +0.0354 |

Tuning selected the best of 50 sampled configurations (CV R2 on log price = 0.7931); the held-out test figure is 0.8265 in log space and 0.7196 on price.
The gap between those two is the selection optimism the notebook's headline omitted.