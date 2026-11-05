from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import pandas as pd

from src.claims import Claim
from src.verifier import evaluate_claim


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "reference_data.parquet"
)


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    return pd.read_parquet(DATASET_PATH)


def get_extractor(provider: str):
    provider = provider.lower().strip()

    if provider == "qwen":
        from src.qwen_extractor import extract_claim
        return extract_claim

    if provider == "gemini":
        from src.gemini_extractor import extract_claim
        return extract_claim

    raise ValueError(
        "Unsupported extractor. Choose 'qwen' or 'gemini'."
    )


def convert_to_claim(extracted) -> Claim:
    if extracted.needs_clarification:
        raise ValueError(
            extracted.clarification_question
            or "The question needs clarification."
        )

    parameters = {}

    if extracted.country:
        parameters["country"] = extracted.country

    if extracted.threshold is not None:
        parameters["threshold"] = extracted.threshold

    if extracted.operator:
        parameters["operator"] = extracted.operator

    if extracted.period:
        parameters["month"] = extracted.period

    return Claim(
        claim_id=f"verityai_{uuid.uuid4().hex[:8]}",
        text="",
        claim_type=extracted.claim_type,
        metric=extracted.metric,
        country=extracted.country,
        threshold=(
            extracted.threshold
            if extracted.threshold is not None
            else 0.0
        ),
        operator=extracted.operator or ">",
        parameters=parameters,
        top_n=extracted.top_n,
    )


def compare(value: float, threshold: float, operator: str) -> bool:
    operations = {
        ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b,
        "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
        "==": lambda a, b: a == b,
    }

    if operator not in operations:
        raise ValueError(f"Unsupported operator: {operator}")

    return operations[operator](value, threshold)


def run_analysis(df: pd.DataFrame, claim: Claim) -> dict:
    claim_type = claim.claim_type

    revenue_column = (
        "line_revenue"
        if "line_revenue" in df.columns
        else "line_value"
    )

    if claim_type == "country_revenue_threshold":
        truth, value, evidence = evaluate_claim(df, claim)

        verdict = (
            "insufficient_evidence"
            if truth is None
            else "supported" if truth else "refuted"
        )

        return {
            "result_type": "claim_verification",
            "observed_value": value,
            "verdict": verdict,
            "evidence": evidence,
        }

    if claim_type == "monthly_revenue":
        month = claim.parameters.get("month")

        valid = df.loc[
            df["invoice_month"].astype(str) == str(month)
        ]

        if valid.empty:
            return {
                "result_type": "analytical_answer",
                "observed_value": None,
                "verdict": "insufficient_evidence",
                "evidence": f"No records found for {month}.",
            }

        value = pd.to_numeric(
            valid[revenue_column], errors="coerce"
        ).sum()

        return {
            "result_type": "analytical_answer",
            "observed_value": float(value),
            "verdict": None,
            "evidence": f"Total revenue for {month}; rows={len(valid)}.",
        }

    if claim_type == "monthly_revenue_threshold":
        valid = df.copy()

        valid[revenue_column] = pd.to_numeric(
            valid[revenue_column], errors="coerce"
        )

        valid = valid.dropna(
            subset=["invoice_month", revenue_column]
        )

        monthly = valid.groupby("invoice_month")[revenue_column].sum()

        if monthly.empty:
            return {
                "result_type": "claim_verification",
                "observed_value": None,
                "verdict": "insufficient_evidence",
                "evidence": "No valid monthly revenue records.",
            }

        month = str(monthly.idxmax())
        maximum = float(monthly.max())

        truth = compare(
            maximum,
            float(claim.threshold),
            claim.operator,
        )

        return {
            "result_type": "claim_verification",
            "observed_value": maximum,
            "verdict": "supported" if truth else "refuted",
            "evidence": (
                f"Highest monthly revenue={maximum:.2f}; "
                f"month={month}; threshold={claim.threshold}; "
                f"operator={claim.operator}."
            ),
        }

    if claim_type == "top_country_revenue":
        valid = df[["country", revenue_column]].copy()

        valid[revenue_column] = pd.to_numeric(
            valid[revenue_column], errors="coerce"
        )

        valid = valid.dropna(
            subset=["country", revenue_column]
        )

        if valid.empty:
            return {
                "result_type": "analytical_answer",
                "observed_value": None,
                "verdict": "insufficient_evidence",
                "evidence": "No valid country revenue records.",
            }

        totals = (
            valid.groupby("country")[revenue_column]
            .sum()
            .sort_values(ascending=False)
        )

        top_n = max(1, min(int(claim.top_n), 20))
        top_countries = [
            {"country": str(country), "revenue": float(revenue)}
            for country, revenue in totals.head(top_n).items()
        ]

        return {
            "result_type": "analytical_answer",
            "observed_value": top_countries if top_n > 1 else top_countries[0],
            "verdict": None,
            "evidence": (
                f"Top {len(top_countries)} countries ranked by total revenue "
                f"out of {len(totals)} countries."
            ),
        }

    if claim_type == "average_unit_price":
        if "unit_price" not in df.columns:
            raise ValueError("Dataset has no unit_price column.")

        prices = pd.to_numeric(
            df["unit_price"], errors="coerce"
        ).dropna()

        if prices.empty:
            return {
                "result_type": "analytical_answer",
                "observed_value": None,
                "verdict": "insufficient_evidence",
                "evidence": "No valid unit prices.",
            }

        value = float(prices.mean())

        return {
            "result_type": "analytical_answer",
            "observed_value": value,
            "verdict": None,
            "evidence": f"Average of {len(prices):,} valid unit prices.",
        }

    if claim_type == "row_count":
        return {
            "result_type": "analytical_answer",
            "observed_value": len(df),
            "verdict": None,
            "evidence": "Count of transaction records in the dataset.",
        }

    raise ValueError(f"Unsupported claim type: {claim_type}")


def verify_question(question: str, provider: str | None = None) -> dict:
    provider = (
        provider or os.getenv("AI DATAAUDIT_EXTRACTOR", "qwen")
    ).lower().strip()

    print(f"\n[1/4] Extracting claim with {provider.upper()}...")

    extractor = get_extractor(provider)
    extracted = extractor(question)

    if extracted.needs_clarification:
        return {
            "original_question": question,
            "extractor": provider,
            "verdict": "needs_clarification",
            "clarification_question": extracted.clarification_question,
        }

    print("[2/4] Creating structured claim...")
    claim = convert_to_claim(extracted)

    print("[3/4] Loading reference dataset...")
    df = load_dataset()
    print(f"      Loaded {len(df):,} records.")

    print("[4/4] Running deterministic analytical engine...")
    analysis = run_analysis(df, claim)

    return {
        "original_question": question,
        "extractor": provider,
        "extracted_claim": extracted.model_dump(),
        "claim_id": claim.claim_id,
        "dataset_rows": len(df),
        "verification_method": "deterministic_python",
        "llm_used_for_verdict": False,
        **analysis,
    }


def main():
    print("=" * 65)
    print("                 AI DATAAUDIT CLAIM VERIFICATION")
    print("=" * 65)

    default_provider = os.getenv("AI DATAAUDIT_EXTRACTOR", "qwen")
    print(f"Extractor: {default_provider.upper()}")
    print("To use Gemini instead, set AI DATAAUDIT_EXTRACTOR=gemini.")

    question = input("\nEnter your question: ").strip()

    if not question:
        print("Please enter a question.")
        return

    try:
        result = verify_question(question)

        print("\n" + "=" * 65)
        print("                    AI DATAAUDIT RESULT")
        print("=" * 65)
        print(json.dumps(result, indent=2, default=str))

        if result.get("verdict"):
            print("\nFINAL VERDICT:", result["verdict"].upper())

        if result.get("observed_value") is not None:
            print("\nCALCULATED VALUE:", result["observed_value"])

    except Exception as exc:
        print(f"\nERROR: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
