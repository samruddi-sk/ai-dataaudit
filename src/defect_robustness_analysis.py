from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "results" / "seed_experiment_results.csv"
RESULTS_DIR = ROOT / "results"

DETAIL_FILE = RESULTS_DIR / "defect_robustness_results.csv"
SUMMARY_FILE = RESULTS_DIR / "defect_robustness_summary.csv"


def numeric_deviation(observed, reference):
    observed = pd.to_numeric(observed, errors="coerce")
    reference = pd.to_numeric(reference, errors="coerce")

    if pd.isna(observed) or pd.isna(reference):
        return np.nan

    if reference == 0:
        return 0.0 if observed == 0 else np.nan

    return abs(observed - reference) / abs(reference) * 100


def main():
    print("=" * 68)
    print("AI DATAAUDIT - DEFECT ROBUSTNESS ANALYSIS")
    print("=" * 68)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Seed experiment results not found. "
            "Run python3 -m src.run_seed_experiments first."
        )

    df = pd.read_csv(INPUT_FILE)

    required = {
        "claim_id",
        "dataset_version",
        "experiment_label",
        "defect_type",
        "repair_applied",
        "verdict",
        "observed_value",
        "reference_value",
        "seed",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")

    df["observed_value"] = pd.to_numeric(
        df["observed_value"], errors="coerce"
    )
    df["reference_value"] = pd.to_numeric(
        df["reference_value"], errors="coerce"
    )

    df["absolute_deviation"] = (
        df["observed_value"] - df["reference_value"]
    ).abs()

    df["relative_deviation_pct"] = df.apply(
        lambda row: numeric_deviation(
            row["observed_value"],
            row["reference_value"],
        ),
        axis=1,
    )

    # A 1% threshold is used only as an exploratory flag.
    df["material_deviation_1pct"] = (
        df["relative_deviation_pct"] >= 1.0
    )

    # Compare defective and repaired versions for each
    # experiment, seed and claim.
    comparison_keys = [
        "seed",
        "experiment_label",
        "defect_type",
        "claim_id",
    ]

    defective = df[
        df["dataset_version"] == "defected"
    ].copy()

    repaired = df[
        df["dataset_version"] == "repaired"
    ].copy()

    comparison = defective.merge(
        repaired,
        on=comparison_keys,
        suffixes=("_defected", "_repaired"),
        how="inner",
        validate="one_to_one",
    )

    comparison["deviation_reduction_pct"] = (
        comparison["relative_deviation_pct_defected"]
        - comparison["relative_deviation_pct_repaired"]
    )

    comparison["repair_improved"] = (
        comparison["deviation_reduction_pct"] > 0
    )

    comparison["repair_worsened"] = (
        comparison["deviation_reduction_pct"] < 0
    )

    comparison["verdict_changed_after_repair"] = (
        comparison["verdict_defected"]
        != comparison["verdict_repaired"]
    )

    comparison["defected_verdict_matches_reference"] = (
        comparison["verdict_defected"] == "supported"
    )

    comparison["repaired_verdict_matches_reference"] = (
        comparison["verdict_repaired"] == "supported"
    )

    comparison["reference_claim_verdict"] = "supported"

    # Per-defect summary.
    summary = comparison.groupby(
        ["defect_type", "experiment_label"],
        dropna=False,
    ).agg(
        seeds=("seed", "nunique"),
        claim_comparisons=("claim_id", "count"),
        mean_defected_deviation_pct=(
            "relative_deviation_pct_defected", "mean"
        ),
        median_defected_deviation_pct=(
            "relative_deviation_pct_defected", "median"
        ),
        mean_repaired_deviation_pct=(
            "relative_deviation_pct_repaired", "mean"
        ),
        median_repaired_deviation_pct=(
            "relative_deviation_pct_repaired", "median"
        ),
        material_defected_cases=(
            "material_deviation_1pct_defected", "sum"
        ),
        material_repaired_cases=(
            "material_deviation_1pct_repaired", "sum"
        ),
        repair_improved_cases=("repair_improved", "sum"),
        repair_worsened_cases=("repair_worsened", "sum"),
        verdict_changes=("verdict_changed_after_repair", "sum"),
    ).reset_index()

    summary["mean_deviation_reduction_pct"] = (
        summary["mean_defected_deviation_pct"]
        - summary["mean_repaired_deviation_pct"]
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(DETAIL_FILE, index=False)
    summary.to_csv(SUMMARY_FILE, index=False)

    print("\nOVERALL RESULTS")

    print(f"Input records: {len(df)}")
    print(f"Defected/repaired comparisons: {len(comparison)}")
    print(f"Defect conditions: {comparison['defect_type'].nunique()}")
    print(f"Random seeds: {comparison['seed'].nunique()}")

    print("\nDEFECT-WISE ANALYSIS")

    columns = [
        "defect_type",
        "mean_defected_deviation_pct",
        "mean_repaired_deviation_pct",
        "mean_deviation_reduction_pct",
        "repair_improved_cases",
        "repair_worsened_cases",
        "verdict_changes",
    ]

    print(
        summary[columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nINTERPRETATION")

    print(
        "Positive deviation reduction means the repaired result "
        "moved closer to the reference value."
    )
    print(
        "Negative deviation reduction means the repair made the "
        "numerical deviation worse."
    )
    print(
        "A verdict change is not automatically an improvement. "
        "Interpret it alongside the reference result."
    )
    print(
        "The 1% material-deviation flag is exploratory, not a "
        "universal definition of materiality."
    )

    print("\nSAVED FILES")
    print(DETAIL_FILE)
    print(SUMMARY_FILE)


if __name__ == "__main__":
    main()
