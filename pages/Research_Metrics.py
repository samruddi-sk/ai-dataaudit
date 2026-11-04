import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

st.set_page_config(
    page_title="VerityAI Research Metrics",
    page_icon="📊",
    layout="wide"
)

st.title("VerityAI | Research Metrics Dashboard")
st.caption(
    "Experimental evaluation of claim extraction, "
    "synthetic verification tests and data-defect robustness."
)

st.divider()


def load_json(filename):
    path = RESULTS / filename
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as error:
        st.warning(f"Could not read {filename}: {error}")
        return None


def load_csv(filename):
    path = RESULTS / filename
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as error:
        st.warning(f"Could not read {filename}: {error}")
        return None


def find_metric(data, possible_names):
    """Find a metric recursively without assuming a fixed JSON layout."""
    if isinstance(data, dict):
        for key, value in data.items():
            if key.lower() in possible_names and isinstance(value, (int, float)):
                return value
        for value in data.values():
            result = find_metric(value, possible_names)
            if result is not None:
                return result
    elif isinstance(data, list):
        for item in data:
            result = find_metric(item, possible_names)
            if result is not None:
                return result
    return None


def show_metric(label, value, suffix=""):
    if value is None:
        st.metric(label, "Not available")
    else:
        st.metric(label, f"{value}{suffix}")


# --------------------------------------------------
# 1. FINE-TUNING EVALUATION
# --------------------------------------------------

st.header("1. Fine-tuning evaluation")

lora = load_json("lora_evaluation.json")

if lora:
    base_valid = find_metric(lora, {"valid_json", "valid_json_count"})
    base_accuracy = find_metric(lora, {"claim_type_accuracy"})
    base_exact = find_metric(lora, {"exact_match_accuracy"})

    st.markdown("#### Qwen2.5-0.5B-Instruct + LoRA")

    st.info(
        "The evaluation uses synthetic, template-based examples. "
        "These metrics measure performance on this evaluation set, "
        "not real-world generalization."
    )

    st.json(lora, expanded=False)

    st.caption(
        "Raw evaluation results are shown above. Metric values are "
        "not inferred if the JSON structure does not expose them."
    )
else:
    st.warning("results/lora_evaluation.json was not found.")

st.divider()


# --------------------------------------------------
# 2. SYNTHETIC GROUND-TRUTH EVALUATION
# --------------------------------------------------

st.header("2. Synthetic ground-truth evaluation")

ground = load_json("expanded_ground_truth_results.json")
ground_csv = load_csv("expanded_ground_truth_results.csv")

if ground_csv is not None:
    st.write("Ground-truth test records")
    st.dataframe(ground_csv, use_container_width=True, hide_index=True)

    st.metric("Test cases", len(ground_csv))

    verdict_column = next(
        (column for column in ["predicted_verdict", "verdict", "actual_verdict"]
         if column in ground_csv.columns),
        None
    )

    if verdict_column:
        st.markdown("#### Verdict distribution")
        counts = ground_csv[verdict_column].value_counts()
        st.bar_chart(counts)

    correct_column = next(
        (column for column in ["correct", "is_correct", "match"]
         if column in ground_csv.columns),
        None
    )

    if correct_column:
        correct = ground_csv[correct_column].astype(str).str.lower().isin(
            ["true", "1", "yes"]
        ).sum()
        st.metric("Correct cases", f"{correct}/{len(ground_csv)}")
elif ground:
    st.json(ground, expanded=False)
else:
    st.warning("Ground-truth evaluation files were not found.")

st.caption(
    "The 40-case evaluation is a constructed synthetic test suite. "
    "Its results should not be presented as independent real-world accuracy."
)

st.divider()


# --------------------------------------------------
# 3. MULTI-SEED EXPERIMENTS
# --------------------------------------------------

st.header("3. Multi-seed robustness experiments")

seed_df = load_csv("seed_experiment_results.csv")

if seed_df is not None:
    st.write("Experiment records")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total experiment records", len(seed_df))

    with col2:
        if "seed" in seed_df.columns:
            st.metric("Random seeds", seed_df["seed"].nunique())
        else:
            st.metric("Random seeds", "Not available")

    with col3:
        if "defect_type" in seed_df.columns:
            st.metric("Defect conditions", seed_df["defect_type"].nunique())
        else:
            st.metric("Defect conditions", "Not available")

    if "verdict" in seed_df.columns:
        st.markdown("#### Verification verdict distribution")
        st.bar_chart(seed_df["verdict"].value_counts())

    if "experiment_label" in seed_df.columns and "verdict" in seed_df.columns:
        st.markdown("#### Verdicts by experiment condition")
        pivot = pd.crosstab(
            seed_df["experiment_label"],
            seed_df["verdict"]
        )
        st.bar_chart(pivot)

    st.markdown("#### Experiment data")
    st.dataframe(seed_df, use_container_width=True, hide_index=True)

else:
    st.warning("results/seed_experiment_results.csv was not found.")

st.divider()


# --------------------------------------------------
# 4. DEFECT ROBUSTNESS ANALYSIS
# --------------------------------------------------

st.header("4. Defect and repair analysis")

robustness_df = load_csv("defect_robustness_summary.csv")

if robustness_df is not None:
    st.dataframe(
        robustness_df,
        use_container_width=True,
        hide_index=True
    )

    numeric_columns = robustness_df.select_dtypes(
        include="number"
    ).columns.tolist()

    if numeric_columns:
        selected_metric = st.selectbox(
            "Select a metric to visualize",
            numeric_columns
        )

        if "defect_type" in robustness_df.columns:
            chart_data = robustness_df.set_index("defect_type")[
                selected_metric
            ]
            st.bar_chart(chart_data)
else:
    st.warning("Defect robustness summary was not found.")

st.divider()


# --------------------------------------------------
# 5. RESEARCH LIMITATIONS
# --------------------------------------------------

st.header("5. Research limitations")

limitations = [
    "The claim-extraction training and evaluation data are synthetic and template-based.",
    "The 40-case ground-truth suite is constructed to test predefined verification rules.",
    "The 1% deviation threshold is exploratory, not a universal definition of materiality.",
    "The dataset contains multiple countries and currencies; aggregate revenue is not a currency-normalized cross-country comparison.",
    "Missing transactions cannot be reconstructed from remaining records alone.",
    "The fine-tuned Qwen model extracts claims; deterministic Python code calculates evidence and produces verification outcomes."
]

for item in limitations:
    st.markdown(f"- {item}")

st.divider()

st.caption(
    "VerityAI | Research prototype | "
    "LLM-based claim extraction + deterministic evidence verification"
)
