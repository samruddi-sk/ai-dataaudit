from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "results" / "seed_experiment_results.csv"
OUTPUT_FILE = ROOT / "results" / "seed_reliability_analysis.csv"
MATERIALITY_THRESHOLD_PCT = 1.0


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing {INPUT_FILE}. Run python3 -m src.run_seed_experiments first."
        )

    df = pd.read_csv(INPUT_FILE)

    required = {
        "claim_id",
        "dataset_version",
        "defect_type",
        "reference_value",
        "observed_value",
        "verdict",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )

    for column in ["reference_value", "observed_value"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    denominator = df["reference_value"].abs()
    df["absolute_deviation"] = (
        df["observed_value"] - df["reference_value"]
    ).abs()

    df["relative_deviation_pct"] = (
        df["absolute_deviation"] / denominator.replace(0, float("nan"))
    ) * 100

    both_zero = (
        df["reference_value"].eq(0)
        & df["observed_value"].eq(0)
    )
    df.loc[both_zero, "relative_deviation_pct"] = 0.0

    df["material_metric_change"] = (
        df["relative_deviation_pct"] >= MATERIALITY_THRESHOLD_PCT
    )

    df["supported_despite_material_change"] = (
        df["material_metric_change"]
        & df["verdict"].eq("supported")
    )

    df.to_csv(OUTPUT_FILE, index=False)

    print("=" * 76)
    print("VERITYAI — MULTI-SEED RELIABILITY ANALYSIS")
    print("=" * 76)
    print(f"Material-change threshold: {MATERIALITY_THRESHOLD_PCT}%")
    print(f"Total evaluation rows: {len(df)}")

    if "seed" in df.columns:
        print(f"Seeds: {sorted(df['seed'].dropna().unique().tolist())}")

    print("\n1. Metric stability by dataset version")
    stability = df.groupby("dataset_version").agg(
        evaluations=("relative_deviation_pct", "count"),
        material_changes=("material_metric_change", "sum"),
        mean_relative_deviation_pct=("relative_deviation_pct", "mean"),
        median_relative_deviation_pct=("relative_deviation_pct", "median"),
    )
    print(stability.round(4).to_string())

    print("\n2. Metric stability by defect and dataset version")
    defect_summary = df.groupby(
        ["defect_type", "dataset_version"]
    ).agg(
        evaluations=("relative_deviation_pct", "count"),
        material_changes=("material_metric_change", "sum"),
        mean_deviation_pct=("relative_deviation_pct", "mean"),
        median_deviation_pct=("relative_deviation_pct", "median"),
    )
    print(defect_summary.round(4).to_string())

    print("\n3. Supported verdicts with at least 1% deviation")
    cases = df[df["supported_despite_material_change"]]

    display_columns = [
        column
        for column in [
            "seed",
            "claim_id",
            "dataset_version",
            "defect_type",
            "repair_applied",
            "reference_value",
            "observed_value",
            "relative_deviation_pct",
            "verdict",
        ]
        if column in df.columns
    ]

    if cases.empty:
        print("No cases found at the exploratory threshold.")
    else:
        print(cases[display_columns].round(4).to_string(index=False))

    print(f"\nDetailed results saved to: {OUTPUT_FILE}")
    print(
        "\nResearch caveat: numerical agreement with the reference does not "
        "prove that the reference is ground truth or that a repair is valid."
    )


if __name__ == "__main__":
    main()
