# Fraud Detection & Model Monitoring

An end-to-end, reproducible fraud-risk ranking project built around the public
ULB credit-card benchmark. It treats fraud as a rare-event review problem and
measures model quality under chronological holdout, fixed review capacity, and
feature/score drift monitoring.

## Project status

**Validated v1 release.** The one-shot chronological test, monitoring report,
and serving artifact are built and verified locally.

## What this project will demonstrate

- Chronological train, validation, and locked test partitions.
- Rare-event metrics that reflect review operations: average precision, recall
  and precision at fixed alert budgets, and alert volume.
- Model selection on validation data only; the locked test partition is
  evaluated once.
- Batch monitoring for feature and score drift, with label-based performance
  monitoring reported separately.
- A review queue and dashboards with clear limits on what the anonymized
  benchmark can support.

Overall accuracy is not used as a success metric. The model output is a
prioritization score, not a claim that a transaction is definitively fraudulent.

## Locked v1 results

The weighted histogram gradient-boosting candidate was selected on validation
average precision (0.7792 versus 0.7723 for balanced logistic regression). On
the later chronological test window, it scored 56,961 transactions with 75
frauds: average precision was 0.7989 and ROC-AUC was 0.9746.

| Review budget | Alerts | Frauds captured | Precision | Recall |
|---|---:|---:|---:|---:|
| 0.10% | 57 | 51 / 75 | 89.5% | 68.0% |
| 0.25% (primary) | 143 | 62 / 75 | 43.4% | 82.7% |
| 0.50% | 285 | 64 / 75 | 22.5% | 85.3% |

Collapsing exact repeated rows in the locked test changed average precision to
0.7959 and primary-budget recall to 82.4%, a small sensitivity for this split.
The primary run retains all rows because the source has no transaction IDs.

The test window contains two four-hour monitoring batches. The later batch's
average precision fell to 0.6503 from 0.8652 in the first batch. Several feature
PSI values exceeded 1.0, while score PSI remained below 0.008 in both batches.
This shows why feature drift and score/performance monitoring are reported
separately; the later batch contains only 21 frauds, and the benchmark spans
only two days.

## Data

The source is OpenML dataset [1597, `creditcard`](https://www.openml.org/d/1597),
which catalogs 284,807 transactions from two days in September 2013, including
492 fraud labels. The dataset has elapsed time, amount, 28 anonymized PCA
components, and the binary target. Raw data are downloaded locally and excluded
from version control; see [`data/README.md`](data/README.md).

## Local setup

```bash
make install
make data-download
make data-audit
make quality
make smoke-release
make experiment-select
make locked-evaluate
make package-release
make verify-artifact
```

Start the dashboard with `make app` or the scoring API with `make api`. The raw
download stays under `data/raw/` and is ignored by Git. The fixed source digest
and audit report are versioned; the raw binary is not.

`make locked-evaluate` is a one-shot action. Its state file is written before
the locked test is scored; do not delete it or rerun that release version.

## Scope and limits

This is a retrospective benchmark exercise, not a production payment control.
The source covers only two days, lacks original transaction and customer
attributes, and does not expose cardholder or merchant identifiers. Monitoring
results therefore describe this dataset's short chronological slices and do
not establish long-term production drift or real-world loss prevention.
