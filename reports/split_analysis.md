# P1 was wrong: the grouped split scores *higher*, and the reason is the finding

## The prediction and the result

`train.py` recorded, before the first run:

> **P1.** The grouped split scores materially WORSE than the notebook's shuffled split.
> 14.3% of rows duplicate another row's feature vector, so a shuffled split lets the model
> memorise. Falsified if the two agree within 0.01 R².

Measured: best R²(log) of **0.8373 grouped** vs **0.8159 shuffled** — the grouped split is
**0.0214 better**. The prediction is falsified, and in the opposite direction to the one
the falsification clause anticipated.

## Why the reasoning was wrong

The prediction assumed duplicate feature vectors are *memorisable copies*: see a row in
training, meet it again in test, recite the answer. That is the standard leakage story and
it is what group-splitting normally protects against.

It does not apply here, because these duplicates **disagree about the price**:

| Measure | Value |
|---|---|
| Multi-member feature groups | 931 |
| Rows sitting in one | 2,451 (23.1%) |
| Groups whose members share an identical price | **0 of 931 (0.0%)** |
| Median within-group price spread | 0.100 crore (**13.3%** of group mean) |

Every duplicate pair is two listings that look identical to the model and sold for
different amounts. They are not copies of an answer — they are **contradictory labels**.

## What that does to each split

| | Shuffled | Grouped |
|---|---|---|
| Test rows | 2,125 | 2,126 |
| Test rows in a multi-member group | 23.6% | 23.3% |
| **Test rows whose twin sits in the training half** | **20.9% (444 rows)** | **0.0%** |
| Best-possible mean abs log-error forced on those rows | 0.0654 | — |
| Test-set log-price variance | 0.0983 | 0.1105 |

Under the shuffled split, 444 test rows have a twin in training carrying a *different*
price. The model learns the training twin's price and is then scored against the other
one. It cannot win: a mean absolute log-error of 0.065 is imposed on a fifth of the test
set before the model does anything.

Grouping removes those rows from the boundary entirely. The grouped test set is therefore
**cleaner, not easier in the leakage sense** — it contains no question with two official
answers.

## So which split should be reported?

Both, which is what the pipeline does. They measure different things:

- **Grouped** answers *"how well does this predict a property whose twin the model has
  never seen?"* — the deployment question, and the honest headline.
- **Shuffled** answers the same question but with a fifth of the test set poisoned by
  label contradiction, so it understates the model by roughly 0.02 R².

The original notebook's shuffled split was therefore **pessimistic here, not optimistic**.
The audit's original framing — that the shuffled split inflated the result — was wrong,
and the correction is recorded in `reports/data_audit.md` and in the `train.py` docstring
that carried the claim.

## The ceiling this implies

Predicting each group's mean log price — the best any model can do without new features —
gives an R²(log) ceiling of **0.9908** overall, and **0.9465** on duplicate rows alone.
The tuned model reaches 0.8265 on held-out data, so roughly 16 points of the remaining gap
is model error rather than label noise. Label noise is not the binding constraint; feature
poverty is (20.9% of rows have `location == "other"`).

## Two other results worth stating

**Tuning made the model slightly worse on held-out data.** The default XGBoost scored
R²(log) **0.8373** on the leaderboard; the 50-draw randomised search selected a
configuration scoring **0.8265** on the same test set — 0.011 lower. Cross-validation
preferred it (GroupKFold CV 0.7931), the test set did not. That gap is within run-to-run
noise, so the defensible claim is *"tuning did not improve held-out performance"*, not
*"tuning improved the model"*.

**SVR is the concrete argument for reporting both metric spaces.** It scores R²(log)
**0.7112** — respectable, mid-table — and R²(price) **−0.4671**, which is worse than
predicting a constant. Exponentiating a slightly biased log prediction explodes on the
expensive tail. Any report quoting only log-space R² would rank SVR above linear
regression; on the scale anyone cares about it is the worst model tested.
