from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "results" / "seed_experiment_results.csv"
OUTPUT_FILE = ROOT / "results" / "seed_experiment_audit.csv"


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

    # When both values are zero, deviation is zero.
    both_zero = (
        df["reference_value"].eq(0)
        & df["observed_value"].eq(0)
    )
    df.loc[both_zero, "relative_deviation_pct"] = 0.0

    df["material_metric_change"] = (
        df["relative_deviation_pct"] >= 1.0
    )
    df["supported_despite_material_change"] = (
        df["material_metric_change"]
        & df["verdict"].eq("supported")
    )

    df.to_csv(OUTPUT_FILE, index=False)

    print("=" * 76)
    print("VERITYAI — MULTI-SEED RESULTS AUDIT")
    print("=" * 76)
    print(f"Input: {INPUT_FILE}")
    print(f"Total evaluation rows: {len(df)}")

    if "seed" in df.columns:
        print(f"Seeds: {sorted(df['seed'].dropna().unique().tolist())}")

    print("\n1. Evaluation counts by dataset version")
    print(df.groupby("dataset_version").size().to_string())

    print("\n2. Verdict counts")
    print(
        df.groupby(["dataset_version", "verdict"])
        .size()
        .unstack(fill_value=0)
        .to_string()
    )

    print("\n3. Material metric deviations (at least 1%)")
    summary = df.groupby("dataset_version").agg(
        evaluations=("material_metric_change", "size"),
        material_changes=("material_metric_change", "sum"),
        mean_deviation_pct=("relative_deviation_pct", "mean"),
        median_deviation_pct=("relative_deviation_pct", "median"),
    )
    print(summary.round(4).to_string())

    print("\n4. Supported verdicts despite material deviations")
    cases = df[df["supported_despite_material_change"]]

    display_columns = [
        column
        for column in [
            "seed",
            "claim_id",
            "dataset_version",
            "defect_type",
            "reference_value",
            "observed_value",
            "relative_deviation_pct",
            "verdict",
        ]
        if column in df.columns
    ]

    if cases.empty:
        print("No cases found at the exploratory 1% threshold.")
    else:
        print(cases[display_columns].round(4).to_string(index=False))

    print(f"\nDetailed audit saved to: {OUTPUT_FILE}")
    print(
        "\nCaveat: 1% is an exploratory diagnostic threshold, not a "
        "validated universal definition of materiality."
    )


if __name__ == "__main__":
    main()
