from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import pandas as pd

from src.claims import Claim


SUPPORTED_VERDICTS = {
    "supported",
    "refuted",
    "insufficient_evidence",
    "unreliable_due_to_data_defects",
}


@dataclass
class VerificationResult:
    claim_id: str
    verdict: str
    observed_value: Any
    reference_value: Any
    evidence: str
    repair_sensitivity: float = 0.0
    confidence: float = 0.5
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _revenue_column(df: pd.DataFrame) -> str:
    if "line_revenue" in df.columns:
        return "line_revenue"
    if "line_value" in df.columns:
        return "line_value"
    raise ValueError(
        "Dataset requires 'line_revenue' or 'line_value'."
    )


def _compare(value: float, threshold: float, operator: str) -> bool:
    if operator == ">":
        return value > threshold
    if operator == ">=":
        return value >= threshold
    if operator == "<":
        return value < threshold
    if operator == "<=":
        return value <= threshold
    if operator == "==":
        return value == threshold
    raise ValueError(f"Unsupported operator: {operator}")


def _get_tolerance(claim: Claim) -> float:
    """Read tolerance consistently from parameters or the legacy field."""
    value = claim.parameters.get("tolerance", claim.tolerance)
    tolerance = float(value)

    if not 0 <= tolerance <= 1:
        raise ValueError("Claim tolerance must be between 0 and 1.")

    return tolerance


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce").dropna()


def evaluate_claim(
    df: pd.DataFrame,
    claim: Claim,
) -> tuple[bool | None, float | None, str]:
    """Evaluate a supported claim and return truth, value, and evidence."""
    template = claim.template_id or claim.claim_type

    if df.empty:
        return None, None, "Dataset is empty."

    tolerance = _get_tolerance(claim)

    if template in {
        "country_revenue_threshold",
        "monthly_revenue",
        "monthly_revenue_threshold",
        "top_country_revenue",
    }:
        revenue_col = _revenue_column(df)
        revenue = _numeric_series(df, revenue_col)

        if revenue.empty:
            return None, None, "No valid revenue values."

    # Country revenue threshold.
    if template == "country_revenue_threshold":
        if not claim.country or "country" not in df.columns:
            return None, None, "Country information is missing."

        mask = df["country"].astype(str) == str(claim.country)
        subset = df.loc[mask]

        if len(subset) < 5:
            return None, None, (
                "Too few rows for a reliable country-level result."
            )

        values = _numeric_series(subset, revenue_col)
        if values.empty:
            return None, None, "No valid revenue values for this country."

        value = float(values.sum())
        truth = _compare(
            value,
            float(claim.threshold),
            claim.operator,
        )

        return truth, value, (
            f"Country={claim.country}; rows={len(subset)}; "
            f"revenue={value:.2f}; threshold={claim.threshold:.2f}; "
            f"operator={claim.operator}"
        )

    # Monthly revenue compared with a reference-derived expected value.
    if template == "monthly_revenue":
        month = claim.parameters.get("month")

        if month is None or "invoice_month" not in df.columns:
            return None, None, "Month information is missing."

        subset = df.loc[
            df["invoice_month"].astype(str) == str(month)
        ]
        values = _numeric_series(subset, revenue_col)

        if values.empty:
            return None, None, f"No valid revenue values for {month}."

        value = float(values.sum())
        expected = float(claim.parameters["threshold"])

        relative_error = (
            abs(value - expected) / max(abs(expected), 1e-9)
        )
        truth = relative_error <= tolerance

        return truth, value, (
            f"Month={month}; revenue={value:.2f}; "
            f"expected={expected:.2f}; "
            f"relative_error={relative_error:.6f}; "
            f"tolerance={tolerance:.6f}"
        )

    # Check whether any month exceeds a threshold.
    if template == "monthly_revenue_threshold":
        if "invoice_month" not in df.columns:
            return None, None, "Month information is missing."

        valid = df.loc[df["invoice_month"].notna()].copy()
        valid[revenue_col] = pd.to_numeric(
            valid[revenue_col], errors="coerce"
        )
        valid = valid.dropna(subset=[revenue_col])

        if valid.empty:
            return None, None, "No valid monthly revenue data."

        monthly = valid.groupby("invoice_month")[revenue_col].sum()
        value = float(monthly.max())
        threshold = float(claim.parameters["threshold"])

        return value > threshold, value, (
            f"Maximum monthly revenue={value:.2f}; "
            f"threshold={threshold:.2f}"
        )

    # Highest-revenue country.
    if template == "top_country_revenue":
        if "country" not in df.columns:
            return None, None, "Country information is missing."

        valid = df[["country", revenue_col]].copy()
        valid[revenue_col] = pd.to_numeric(
            valid[revenue_col], errors="coerce"
        )
        valid = valid.dropna(subset=["country", revenue_col])

        if valid.empty:
            return None, None, "No valid country revenue data."

        totals = valid.groupby("country")[revenue_col].sum()
        actual_country = str(totals.idxmax())
        expected_country = str(
            claim.parameters.get("country", claim.country)
        )

        return actual_country == expected_country, float(totals.max()), (
            f"Highest-revenue country={actual_country}; "
            f"expected={expected_country}"
        )

    # Average unit price.
    if template == "average_unit_price":
        if "unit_price" not in df.columns:
            return None, None, "Unit-price data is unavailable."

        prices = _numeric_series(df, "unit_price")
        if prices.empty:
            return None, None, "No valid unit prices."

        value = float(prices.mean())
        expected = float(claim.parameters["threshold"])

        relative_error = (
            abs(value - expected) / max(abs(expected), 1e-9)
        )
        truth = relative_error <= tolerance

        return truth, value, (
            f"Average unit price={value:.4f}; "
            f"expected={expected:.4f}; "
            f"relative_error={relative_error:.6f}; "
            f"tolerance={tolerance:.6f}"
        )

    # Dataset row count.
    if template == "row_count":
        value = float(len(df))
        expected = float(claim.parameters["threshold"])

        return value == expected, value, (
            f"Observed rows={int(value)}; expected rows={int(expected)}"
        )

    return None, None, f"Unsupported claim template: {template}"


def _relative_shift(reference_value: Any, observed_value: Any) -> float | None:
    """Return relative numerical shift, or None for unusable values."""
    if reference_value is None or observed_value is None:
        return None

    try:
        reference = float(reference_value)
        observed = float(observed_value)
    except (TypeError, ValueError):
        return None

    if not pd.notna(reference) or not pd.notna(observed):
        return None

    if reference == 0:
        return 0.0 if observed == 0 else None

    return abs(observed - reference) / abs(reference)


def verify_pair(
    reference_df: pd.DataFrame,
    defected_df_or_claim,
    claim: Claim | None = None,
    repaired_variants: list[pd.DataFrame] | None = None,
) -> VerificationResult:
    """
    Compare a claim evaluated on a reference dataset and an observed dataset.

    This is reference-relative verification, not independent fact verification.
    """
    if claim is None:
        claim = defected_df_or_claim
        defected_df = reference_df
    else:
        defected_df = defected_df_or_claim

    reference_truth, reference_value, reference_evidence = evaluate_claim(
        reference_df, claim
    )
    observed_truth, observed_value, observed_evidence = evaluate_claim(
        defected_df, claim
    )

    tolerance = _get_tolerance(claim)
    shift = _relative_shift(reference_value, observed_value)

    if reference_truth is None:
        verdict = "insufficient_evidence"
        notes = "The reference evidence is insufficient."
    elif observed_truth is None:
        verdict = "insufficient_evidence"
        notes = "The observed evidence is insufficient."
    elif reference_truth != observed_truth:
        verdict = "unreliable_due_to_data_defects"
        notes = "The claim truth value changed between datasets."
    elif shift is not None and shift > tolerance:
        verdict = "unreliable_due_to_data_defects"
        notes = (
            "The numerical result shifted beyond the claim tolerance, "
            "although the Boolean claim result remained unchanged."
        )
    else:
        verdict = "supported" if observed_truth else "refuted"
        notes = (
            "Observed claim result matches the reference within tolerance. "
            "This does not independently establish factual correctness."
        )

    changes = 0
    valid_variants = 0

    for variant in repaired_variants or []:
        variant_truth, variant_value, _ = evaluate_claim(variant, claim)

        if variant_truth is None:
            continue

        valid_variants += 1

        if reference_truth is not None and variant_truth != reference_truth:
            changes += 1
            continue

        variant_shift = _relative_shift(reference_value, variant_value)
        if variant_shift is not None and variant_shift > tolerance:
            changes += 1

    sensitivity = (
        changes / valid_variants if valid_variants else 0.0
    )

    confidence = (
        0.9
        if verdict in {"supported", "refuted"}
        else 0.4
    )

    return VerificationResult(
        claim_id=claim.claim_id,
        verdict=verdict,
        observed_value=observed_value,
        reference_value=reference_value,
        evidence=(
            f"Reference: {reference_evidence}\n"
            f"Observed: {observed_evidence}"
        ),
        repair_sensitivity=sensitivity,
        confidence=confidence,
        notes=notes,
    )
