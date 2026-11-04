from pathlib import Path
import sys

import streamlit as st
import pandas as pd

from src.verify_user_claim import (
    verify_question,
    load_dataset,
    convert_to_claim,
    run_analysis,
)
from src.defects import inject_defect


ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="VerityAI | Claim Verification",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: 750;
        letter-spacing: -1.5px;
    }
    .subtitle {
        color: #9CA3AF;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .result-card {
        padding: 20px;
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 12px;
        margin: 12px 0;
    }
    .small-label {
        color: #9CA3AF;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.2);
        padding: 15px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading reference dataset...")
def get_dataset():
    return load_dataset()


def display_value(value):
    if value is None:
        return "Not available"

    if isinstance(value, dict):
        return " | ".join(
            f"{key}: {val:,.2f}" if isinstance(val, float)
            else f"{key}: {val}"
            for key, val in value.items()
        )

    if isinstance(value, (int, float)):
        return f"{value:,.4f}"

    return str(value)


def show_verdict(verdict):
    labels = {
        "supported": ("SUPPORTED", "success"),
        "refuted": ("REFUTED", "error"),
        "insufficient_evidence": ("INSUFFICIENT EVIDENCE", "warning"),
        "unreliable_due_to_data_defects": (
            "UNRELIABLE DUE TO DATA DEFECTS", "warning"
        ),
        "needs_clarification": ("NEEDS CLARIFICATION", "info"),
    }

    label, level = labels.get(
        verdict, (str(verdict).upper(), "info")
    )

    if level == "success":
        st.success(f"VERDICT: {label}")
    elif level == "error":
        st.error(f"VERDICT: {label}")
    elif level == "warning":
        st.warning(f"VERDICT: {label}")
    else:
        st.info(f"VERDICT: {label}")


# Sidebar
with st.sidebar:
    st.markdown("## 🔎 VerityAI")
    st.caption("Research prototype | AI claim verification")
    st.divider()

    st.markdown("### Model architecture")
    st.markdown("**Claim extraction**")
    st.caption("Qwen2.5-0.5B-Instruct + LoRA")

    st.markdown("**Evidence computation**")
    st.caption("Python + Pandas")

    st.markdown("**Verdict generation**")
    st.caption("Deterministic verification engine")

    st.divider()
    st.markdown("### Research dataset")
    st.caption("UCI Online Retail II")
    st.caption("1,007,913 transaction records")
    st.caption("Multiple countries and currencies")

    st.divider()
    st.caption("LLM extracts claims; it does not decide factual verdicts.")
    st.caption("VerityAI | Research Prototype")


# Header
st.markdown(
    '<div class="main-title">VerityAI</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">'
    'Beyond query execution: repair-robust verification of '
    'AI-generated claims under data quality defects.'
    '</div>',
    unsafe_allow_html=True,
)

try:
    df = get_dataset()
except Exception as exc:
    st.error(f"Could not load the reference dataset: {exc}")
    st.info("Check that data/processed/reference_data.parquet exists.")
    st.stop()


# Dataset overview
c1, c2, c3 = st.columns(3)
c1.metric("Reference records", f"{len(df):,}")
c2.metric("Countries", f"{df['country'].nunique():,}" if "country" in df else "N/A")
c3.metric("Verification engine", "Deterministic")

st.divider()

tab1, tab2, tab3 = st.tabs([
    "Claim Verification",
    "Data Robustness",
    "About the Research",
])


# TAB 1: CLAIM VERIFICATION
with tab1:
    st.subheader("Verify a natural-language claim")
    st.write(
        "Enter a question about the retail dataset. "
        "Qwen extracts its structure and Python independently "
        "calculates the evidence."
    )

    question = st.text_area(
        "Your claim",
        placeholder="Example: Was Germany's revenue greater than 500000?",
        height=100,
        key="claim_question",
    )

    examples = [
        "Was Germany's revenue greater than 500000?",
        "What was the revenue in December 2010?",
        "Did any month generate revenue greater than 1000000?",
        "Which country generated the highest revenue?",
        "What was the average unit price?",
        "How many transaction records are there?",
    ]

    st.caption("Try an example:")
    example_cols = st.columns(3)

    for index, example in enumerate(examples):
        with example_cols[index % 3]:
            if st.button(
                example,
                key=f"example_{index}",
                use_container_width=True,
            ):
                st.session_state["claim_question"] = example
                st.rerun()

    if st.button(
        "Verify claim",
        type="primary",
        use_container_width=True,
        key="verify_button",
    ):
        if not question.strip():
            st.warning("Enter a claim first.")
        else:
            try:
                with st.spinner(
                    "Qwen is extracting the claim and the "
                    "verification engine is calculating evidence..."
                ):
                    result = verify_question(
                        question.strip(),
                        provider="qwen",
                    )

                st.session_state["last_result"] = result

            except Exception as exc:
                st.error(f"Verification failed: {type(exc).__name__}: {exc}")

    result = st.session_state.get("last_result")

    if result:
        st.divider()
        st.subheader("Verification result")

        if result.get("verdict") == "needs_clarification":
            show_verdict("needs_clarification")
            st.write(result.get("clarification_question"))
        else:
            verdict = result.get("verdict")
            if verdict:
                show_verdict(verdict)
            else:
                st.info("ANALYTICAL ANSWER")

            extracted = result.get("extracted_claim", {})

            left, right = st.columns(2)

            with left:
                st.markdown("**Extracted structured claim**")
                st.json(extracted)

            with right:
                st.markdown("**Calculated observation**")
                st.markdown(
                    f'<div class="result-card">'
                    f'<div class="small-label">Observed value</div>'
                    f'<h3>{display_value(result.get("observed_value"))}</h3>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                st.markdown("**Evidence**")
                st.write(result.get("evidence", "No evidence available."))

            with st.expander("Technical verification details"):
                st.write("**Extractor:**", result.get("extractor"))
                st.write(
                    "**Verification method:**",
                    result.get("verification_method"),
                )
                st.write(
                    "**LLM used for verdict:**",
                    result.get("llm_used_for_verdict"),
                )
                st.write("**Dataset rows:**", result.get("dataset_rows"))
                st.write("**Claim ID:**", result.get("claim_id"))

            st.download_button(
                "Download verification result",
                data=pd.Series(result).to_json(indent=2),
                file_name="verityai_verification.json",
                mime="application/json",
            )


# TAB 2: DATA ROBUSTNESS
with tab2:
    st.subheader("Data defect robustness experiment")
    st.write(
        "Evaluate how a selected data quality defect changes "
        "the evidence produced for a natural-language claim."
    )

    robustness_question = st.text_input(
        "Claim for robustness testing",
        value="Was Germany's revenue greater than 500000?",
        key="robustness_question",
    )

    defect_options = {
        "Duplicate transactions": "duplicate_rows",
        "Missing customer IDs": "missing_customer_ids",
        "Invalid prices": "invalid_prices",
        "Missing transactions": "missing_transactions",
    }

    selected_label = st.selectbox(
        "Select a simulated data defect",
        list(defect_options.keys()),
    )

    seed = st.number_input(
        "Random seed",
        min_value=0,
        value=42,
        step=1,
    )

    if st.button(
        "Run robustness experiment",
        type="primary",
        use_container_width=True,
    ):
        if not robustness_question.strip():
            st.warning("Enter a claim first.")
        else:
            try:
                with st.spinner("Running defect experiment..."):
                    from src.qwen_extractor import extract_claim

                    extracted = extract_claim(robustness_question)
                    claim = convert_to_claim(extracted)

                    defected_df, defect_info = inject_defect(
                        df,
                        defect_options[selected_label],
                        seed=int(seed),
                    )

                    reference_result = run_analysis(df, claim)
                    defected_result = run_analysis(defected_df, claim)

                    reference_value = reference_result.get("observed_value")
                    defected_value = defected_result.get("observed_value")

                    reference_verdict = reference_result.get("verdict")
                    defected_verdict = defected_result.get("verdict")

                    if reference_value is None or defected_value is None:
                        deviation = None
                    elif isinstance(reference_value, (int, float)) and isinstance(
                        defected_value, (int, float)
                    ):
                        deviation = (
                            abs(defected_value - reference_value)
                            / max(abs(reference_value), 1e-9)
                        ) * 100
                    elif isinstance(reference_value, dict) and isinstance(
                        defected_value, dict
                    ):
                        if (
                            reference_value.get("country")
                            == defected_value.get("country")
                        ):
                            a = reference_value.get("revenue")
                            b = defected_value.get("revenue")
                            deviation = (
                                abs(b - a) / max(abs(a), 1e-9)
                            ) * 100
                        else:
                            deviation = None
                    else:
                        deviation = None

                    if (
                        reference_verdict is not None
                        and defected_verdict is not None
                        and reference_verdict != defected_verdict
                    ):
                        robustness_verdict = "unreliable_due_to_data_defects"
                    elif deviation is not None and deviation > 1:
                        robustness_verdict = "unreliable_due_to_data_defects"
                    elif reference_verdict is not None:
                        robustness_verdict = reference_verdict
                    else:
                        robustness_verdict = "analytical_result"

                    st.session_state["robustness_result"] = {
                        "claim": extracted.model_dump(),
                        "reference": reference_result,
                        "defected": defected_result,
                        "defect_info": defect_info,
                        "deviation": deviation,
                        "robustness_verdict": robustness_verdict,
                    }

            except Exception as exc:
                st.error(f"Experiment failed: {type(exc).__name__}: {exc}")

    experiment = st.session_state.get("robustness_result")

    if experiment:
        st.divider()
        st.subheader("Experiment output")

        show_verdict(experiment["robustness_verdict"])

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Reference observation",
            display_value(experiment["reference"].get("observed_value")),
        )

        col2.metric(
            "Defected observation",
            display_value(experiment["defected"].get("observed_value")),
        )

        deviation = experiment["deviation"]

        col3.metric(
            "Relative deviation",
            "N/A" if deviation is None else f"{deviation:.4f}%",
        )

        st.markdown("**Extracted claim**")
        st.json(experiment["claim"])

        evidence_col1, evidence_col2 = st.columns(2)

        with evidence_col1:
            st.markdown("**Reference evidence**")
            st.write(experiment["reference"].get("evidence"))

        with evidence_col2:
            st.markdown("**Defected-data evidence**")
            st.write(experiment["defected"].get("evidence"))

        with st.expander("Defect metadata"):
            st.json(experiment["defect_info"])

        if defect_options[selected_label] == "missing_transactions":
            st.info(
                "Missing transactions cannot be reconstructed from "
                "the remaining records. This condition represents "
                "an intentional no-repair scenario."
            )

        st.caption(
            "The 1% deviation threshold is an exploratory research "
            "threshold, not a universal definition of materiality."
        )


# TAB 3: ABOUT THE RESEARCH
with tab3:
    st.subheader("Research overview")

    st.markdown("""
    **Research question**

    How reliable are AI-generated claims when the underlying
    structured data contains quality defects, and how do repair
    operations affect verification outcomes?

    **System architecture**

    1. Natural-language input
    2. Fine-tuned Qwen2.5 model extracts a structured claim
    3. Python analytics engine computes evidence
    4. Defect and repair experiments test robustness
    5. VerityAI reports the evidence and outcome

    **Research components**
    """)

    research_cols = st.columns(2)

    with research_cols[0]:
        st.markdown("""
        **Claim extraction**
        - Qwen2.5-0.5B-Instruct
        - LoRA fine-tuning
        - Structured JSON output

        **Analytical verification**
        - Pandas
        - DuckDB
        - Deterministic rules
        """)

    with research_cols[1]:
        st.markdown("""
        **Data quality experiments**
        - Duplicate records
        - Missing customer IDs
        - Invalid prices
        - Missing transactions

        **Evaluation**
        - Multi-seed experiments
        - Synthetic ground-truth tests
        - Robustness and repair analysis
        """)

    st.warning(
        "The current fine-tuning evaluation uses synthetic, "
        "template-based examples. Its reported metrics should not "
        "be interpreted as real-world generalization accuracy."
    )

    st.markdown("**Model evaluation results**")

    evaluation_path = ROOT / "results" / "lora_evaluation.json"

    if evaluation_path.exists():
        try:
            import json

            evaluation = json.loads(evaluation_path.read_text())
            st.json(evaluation)
        except Exception:
            st.caption("Could not read the saved evaluation file.")
    else:
        st.caption("Model evaluation file not found.")

    st.markdown("**Dataset:** UCI Online Retail II")
    st.markdown(
        "The dataset contains transactions across multiple countries. "
        "Monetary totals should be interpreted with the dataset's "
        "multiple-currency limitation in mind."
    )

st.divider()
st.caption(
    "VerityAI | Research prototype | LLM extraction + "
    "deterministic evidence verification"
)
