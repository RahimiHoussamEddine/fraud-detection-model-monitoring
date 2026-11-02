from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Fraud Monitoring", page_icon="🧭", layout="wide")
st.title("Fraud review and monitoring")
st.caption(
    "A short-window, anonymized research benchmark. "
    "Scores prioritize review; they do not prove fraud."
)

metrics_path = Path("reports/locked/v1/locked_metrics.json")
monitoring_path = Path("reports/locked/v1/monitoring_windows.json")
selection_path = Path("artifacts/v1/selection_manifest.json")

if not all(path.is_file() for path in (metrics_path, monitoring_path, selection_path)):
    st.info(
        "The v1 locked evaluation is not available yet. Run the documented pipeline to build it."
    )
    st.stop()

metrics = json.loads(metrics_path.read_text())
monitoring = json.loads(monitoring_path.read_text())
selection = json.loads(selection_path.read_text())
locked = metrics["metrics"]
primary_capacity = locked["capacity_metrics"]["0.0025"]

cards = st.columns(4)
cards[0].metric("Locked test average precision", f"{locked['average_precision']:.3f}")
cards[1].metric("Fraud prevalence", f"{locked['prevalence']:.3%}")
cards[2].metric(
    "Frauds captured at 0.25% review",
    f"{primary_capacity['fraud_captured']} / {primary_capacity['fraud_count']}",
)
cards[3].metric("Review precision at 0.25%", f"{primary_capacity['precision']:.1%}")

st.subheader("Validation model comparison")
comparison = pd.DataFrame(
    [
        {"candidate": name, "validation average precision": result["average_precision"]}
        for name, result in selection["candidate_results"].items()
    ]
).sort_values("validation average precision", ascending=False)
st.plotly_chart(
    px.bar(
        comparison,
        x="candidate",
        y="validation average precision",
        color="candidate",
        title="Candidate ranking quality on validation data",
    ),
    width="stretch",
)
st.caption(f"Selected on validation average precision: `{selection['selected_model']}`.")

st.subheader("Time-window monitoring")
windows = pd.DataFrame(monitoring["windows"])
if windows.empty:
    st.warning("No monitoring windows were produced.")
else:
    drift_rows = []
    for window in monitoring["windows"]:
        scores = window["drift_psi"]
        drift_rows.append(
            {
                "window": window["window_index"],
                "rows": window["rows"],
                "frauds": window["fraud_count"],
                "fraud rate": window["fraud_rate"],
                "score PSI": scores["fraud_score"],
                "max feature PSI": max(
                    value for name, value in scores.items() if name != "fraud_score"
                ),
            }
        )
    chart_data = pd.DataFrame(drift_rows)
    st.plotly_chart(
        px.line(
            chart_data,
            x="window",
            y=["score PSI", "max feature PSI"],
            markers=True,
            title="Distribution drift relative to training data",
        ),
        width="stretch",
    )
    st.dataframe(chart_data, width="stretch", hide_index=True)
    st.caption("PSI 0.10/0.25 thresholds are heuristic review triggers, not universal guarantees.")

st.info(
    "The benchmark covers only two days, contains anonymized PCA features, "
    "and has no customer or card IDs. "
    "These results do not establish long-term production drift or real-world loss reduction."
)
