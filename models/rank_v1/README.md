# rank_v1: move-ranking model (XGBoost rank:pairwise)

- **Trained:** 1 Nov 2026, job J16 on Pat, code `4c7bb9e1` (`benchmark/train_ranker.py`), data J15 (`benchmark/ranking_data.py`).
- **Data:** mech-USPTO-31k gold steps. For each step, the gold AIM move plus 50 random wrong candidates, all with AIM's 182-bit fingerprint (`shared/ml_helper.make_fingerprint`). 80,765 steps / 4.1 M rows.
- **Held out (never trained on):**
  - 8 whole classes: reductive amination, SNAr(para), Mitsunobu, Swern, aldol condensation, Grignard, Wittig, nitrile reduction
  - 10 % of the reactions of every other class
  - the split is defined in `ranking_data.split_of`
- **Output:** a relative score per candidate (higher = more plausible). Only the order within one state means anything; turn scores into probabilities with a softmax over a state's candidates (as `SPArrow/beam_search.py` does).
- **Held-out results** (gold move among all AIM candidates; `results.json`):

  | test set | v121 top-3 | rank_v1 top-3 | rank_v1 top-1 |
  |---|---|---|---|
  | unseen classes (3,937 steps) | 0.440 | **0.680** | 0.404 |
  | unseen reactions, seen classes (2,207 steps) | 0.296 | **0.976** | 0.777 |

- **Use:** `xgboost.XGBRanker().load_model("models/rank_v1/ranker.json")`, then `.predict(fingerprints)`.
