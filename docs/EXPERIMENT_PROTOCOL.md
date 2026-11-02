# Experiment protocol

## Question and unit of analysis

Can a small, reproducible classifier rank the rare fraud examples ahead of
legitimate transactions while respecting a fixed manual-review budget, and can
simple batch monitors describe how the score and feature distributions change
over the short held-out period?

The unit is one transaction row from OpenML `creditcard` (dataset 1597). The
binary target is `Class`; `Time` is elapsed seconds from the first transaction.
`Time` is used for ordering and monitoring only and is excluded from model
features. `Amount` and `V1`–`V28` are the model features.

## Data partitioning

1. Validate the source schema, labels, finite numeric values, amount range, and
   source row/fraud counts.
2. Sort by `Time`, retaining ties together. Split whole timestamp groups into
   the earliest approximately 60% for training, the next approximately 20% for
   validation, and the latest approximately 20% for the locked test.
3. Exact repeated full rows are retained in the primary run because no
   transaction identifier is available to distinguish duplicate ingestion
   from genuine repeated transactions. Locked-test sensitivity metrics will
   also be shown after exact repeated test rows are collapsed. This sensitivity
   does not retrain a separate model.
4. No random split or oversampling is used. The train partition alone fits all
   preprocessing and model parameters.
5. Candidate selection uses validation average precision. Locked test results
   are written once; an evaluation sentinel prevents accidental reruns.

## Candidate and operating policy

- Balanced logistic regression (`lbfgs`, at most 1,000 iterations) with
  train-fitted standardization.
- Histogram gradient boosting (`learning_rate=0.08`, `max_iter=120`,
  `max_leaf_nodes=31`, `l2_regularization=1.0`) with training-only
  inverse-frequency sample weights and internal early stopping disabled.
- Select the highest validation average precision; use model name as a stable
  tie break.
- The selected model, source digest, protocol digest, and validation results are
  frozen before the locked evaluation starts. The locked script writes a
  `started` sentinel before computing test metrics; any interrupted attempt is
  preserved and not rerun under v1.
- The release packaging path is first smoke-checked end to end on synthetic
  data. This exercises selection, the one-shot state machine, artifact creation,
  and inference without touching the real locked test labels.
- Report average precision and ROC-AUC, plus precision and recall in the top
  0.10%, 0.25%, and 0.50% of scored transactions. The 0.25% queue is the
  primary illustrative capacity point; it is not an operational recommendation.
- Equal model scores are ordered with a deterministic seeded tie break. The
  model output is a prioritization score, not a calibrated fraud probability.

## Monitoring protocol

- Compare each four-hour locked-test batch with the training-period feature
  and score distributions.
- Population stability index (PSI) bins are quantiles fitted from training
  data only. The displayed 0.10 watch and 0.25 alert cutoffs are heuristic
  review triggers, not universal statistical guarantees.
- Report label-based batch average precision and fixed-capacity precision and
  recall only when a batch contains both classes. Always show its sample and
  fraud counts.
- Keep distribution drift and label-based model performance as separate
  signals. This two-day dataset cannot establish long-term production drift.

## Reporting

Report exact sample counts, prevalence, candidate validation results, locked
test metrics, capacity counts, time-window drift, and source/model digests.
Do not use overall accuracy as a success metric. Do not claim a real-world loss
reduction, deployment effectiveness, or user-level generalization. No
confidence interval is presented as a long-term uncertainty estimate from this
two-day dataset.
