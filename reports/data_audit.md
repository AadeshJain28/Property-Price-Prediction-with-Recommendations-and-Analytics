# Data audit — Bengaluru property listings

Every figure below is produced by `python -m property_price.audit` and written to
`reports/data_audit.json`. Nothing here is typed by hand.

Source: `data/raw/final_data.csv` — 10,626 rows, 11 columns, the cleaned output of the
original capstone notebooks (`notebooks/data_cleaning_preprocessing.ipynb`,
`notebooks/eda_outlier.ipynb`).

## 1. `price_per_sqft` is the target in disguise

`price` is quoted in crore (10^7 rupees) and `area` in square feet, so:

```
price_per_sqft  ==  price * 1e7 / area
```

Measured across all 10,626 rows: **maximum absolute deviation 0.50, and 100% of rows
agree within 1.0** — the residual is the dataset's rounding to whole rupees, nothing
more. The two columns therefore reconstruct `price` exactly.

`tests/test_leakage_guard.py::test_leaked_model_is_near_perfect_so_the_guard_is_load_bearing`
recomputes the target from those two columns and asserts R² > 0.9999. That is the score
a model would report if these columns were left in — a number that would look like a
strong result and mean nothing.

The original notebook drops `price_per_sqft` and `area` at cell 3, which is correct. The
difference here is that dropping them is now enforced: `features.assert_no_banned_features`
raises `LeakageError`, `Config.validate` refuses to load a config that lists them, and CI
runs the guard as a separate step. A silent drop would let a future edit re-add them
without anyone noticing.

## 2. 14.3% of rows duplicate another row's feature vector

| Measure | Value |
|---|---|
| Exact duplicate rows (all columns) | 0 |
| Rows duplicating another row on features alone | **1,520 (14.3%)** |
| Distinct feature-vector groups | 9,106 |

No two rows are identical, but 1,520 share every modelling feature with some other row
while carrying a different price — the same flat listed twice at different asking prices,
or two units indistinguishable given the columns available.

This matters because the original notebook used `KFold(shuffle=True)` and
`train_test_split(random_state=42)`. Under a shuffled split these near-copies land on both
sides of the boundary, and the model is scored partly on rows whose answer it has already
seen. `data.grouped_train_test_split` uses `GroupShuffleSplit` over the duplicate-feature
group id so a group can never straddle the split.

**The size of that effect is measured, not assumed.** `train.py` runs the full leaderboard
twice — once grouped, once with the notebook's shuffled split — and writes both to
`reports/leaderboard.csv` and `reports/leaderboard_naive_split.csv`, with the gap recorded
in `reports/predictions.md`.

> **Correction.** This section originally asserted that the shuffled split *inflated* the
> result through memorisation. It measured the other way: grouped 0.8373 vs shuffled
> 0.8159. **0 of 931** duplicate groups share a price, so these rows are contradictory
> labels rather than memorisable copies, and a shuffled split poisons 20.9% of its own test
> set. The shuffled split was pessimistic here. Derivation in
> [`reports/split_analysis.md`](split_analysis.md).

## 3. The reported R² is in log space, on a target with skew 6.5

| Measure | Value |
|---|---|
| Skew of `price` | 6.53 |
| Skew of `log1p(price)` | 1.88 |

The notebook fits on `log1p(price)` and reports `cross_val_score(..., scoring='r2')` — so
its headline **0.82 is R² on log price, not on price**. With a target this skewed the two
are materially different: squared error on the raw scale is dominated by a handful of
properties above 10 crore that the log transform compresses.

`evaluate.score` therefore always returns both (`r2_log`, `r2_price`) plus MAE, RMSE and
median APE in rupees. A single R² with no space attached is not a defensible claim.

## 4. The 0.82 was also the maximum of 50 tuning draws

The notebook's best figure comes from `RandomizedSearchCV(n_iter=50).best_score_` — the
largest of 50 noisy cross-validation estimates, which is biased upward by selection. The
rebuilt pipeline tunes on training folds only (`GroupKFold`) and touches the held-out test
set exactly once, at the end. Both numbers are reported so the gap between them is visible.

## 5. Smaller findings

| Finding | Value | Consequence |
|---|---|---|
| Distinct locations | 243 | High-cardinality; `OneHotEncoder(min_frequency=10)` pools the tail |
| Rows with `location == "other"` | 20.9% | A fifth of the data has no usable location signal |
| `built_up_area == area` | 28.6% | Imputed rather than observed for these rows |
| `property_type` balance | 9,697 flats / 929 houses | Houses are 8.7% of the data; per-segment error should be reported |
| `area` range | 1 – **1,306,800** sqft | 5 rows exceed 100,000 sqft — land parcels, not flats |

### The largest listings are land, and they distort absolute error

Five rows have `area` above 100,000 sqft, the largest at **1,306,800 sqft listed at 2
rupees/sqft**. These are land parcels sharing a table with apartments.

They surfaced through a failing test rather than an EDA plot. The leakage test asserted
that reconstructing price from the banned columns lands within 0.01 crore of the truth;
it failed in CI at 0.0386. The data was fine — the constant was wrong. Because
`price_per_sqft` is rounded to whole rupees, the reconstruction error in crore is

```
|price - price_per_sqft x area / 1e7|  <=  0.5 x area / 1e7
```

which **grows with area**: 0.065 crore on the 1.3M sqft plot, versus 0.00005 on a typical
1,000 sqft flat. A flat tolerance was never the right test. The assertion is now the
derived inequality, checked row by row — it holds for all 10,626 rows, is tight on 5,227
of them, and cannot go stale if the data changes.

Relative error on the 10,595 listings under 10,000 sqft stays below **3.2e-04**.

## What was wrong, and what caught it

| Claim | Status | Caught by |
|---|---|---|
| "XGBoost achieving best R² score of 0.82" | Incomplete — true in log space (measured 0.8265 held-out), but only 0.7196 on price | `evaluate.score`, `reports/summary.json` |
| *My* claim that the shuffled split inflated that figure | **Wrong** — it deflated it by 0.021; the duplicates are contradictory labels, not copies | `reports/split_analysis.md` |
| "Tuning improved the model" | **Not supported** — default XGBoost scored 0.8373 held-out, the tuned model 0.8265 | `reports/leaderboard.csv` vs `summary.json` |
| Reporting a single R² | Unsafe — SVR scores 0.7112 in log space and **−0.4671** on price | `reports/leaderboard.csv` |
| Recommender "using cosine similarity" | Was unevaluated — now precision@5 = 0.591, location hit-rate 0.961 | `recommend.PropertyRecommender.evaluate` |
| *My* leakage-test tolerance of 0.01 crore | **Wrong** — a flat bound on an error that scales with `area`; failed CI at 0.0386 | `tests/test_leakage_guard.py`, now asserting the derived bound |
| *My* guess that CI failed on a scikit-learn version mismatch | **Wrong** — it was the tolerance above; the artefact never got loaded | the CI log |

### A dead input in the dashboard

`availability` is one of the eight model features and has two levels in the data
(`Ready To Move`, `Under Construction`). The first version of the Streamlit app hardcoded
it to `"Ready To Move"` and exposed no control, so every prediction silently assumed
ready-to-move regardless of what the user meant — and `area_type` was pinned to the mode
the same way.

Not a modelling error; the trained model is unaffected. But a served model whose inputs
the user cannot set is answering a different question from the one being asked.
`tests/test_schema.py::test_availability_is_a_real_feature_not_a_constant` fails if a live
feature ever collapses to a constant again, and both controls are now in the UI.

### A bug in the guard itself

The first version of `features.build_matrix` ran the leakage check against
`df.columns` — the columns of the *source frame* — rather than against the columns
actually selected as features. Since the raw frame is supposed to carry
`price_per_sqft` and `area` (the audit above is computed from them), the guard rejected
every legitimate call and `train.py` could not run at all.

The check belongs on the **output**, as a post-condition: select `cfg.feature_names`,
then assert the result contains nothing banned. Corrected in `features.py`; the
regression is pinned by
`tests/test_leakage_guard.py::test_build_matrix_accepts_a_raw_frame_and_drops_the_banned_columns`,
with `test_build_matrix_still_raises_when_config_smuggles_a_banned_feature` confirming the
guard is still load-bearing after the fix.

Worth recording rather than quietly patching: a guard that fails closed on everything is
indistinguishable, from the outside, from a guard that works — right up until it blocks
the pipeline. This one failed loudly and immediately, which is the cheap case. The
expensive version of the same mistake is a guard that fails *open*.
