from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "defect_robustness_results.csv"
OUTPUT = ROOT / "results" / "statistical_analysis.csv"


def main():
    df = pd.read_csv(INPUT)

    columns = [
        "defect_type",
        "experiment_label",
        "seed",
        "relative_deviation_pct_defected",
        "relative_deviation_pct_repaired",
        "deviation_reduction_pct",
    ]

    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    for col in columns[2:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=[
            "relative_deviation_pct_defected",
            "relative_deviation_pct_repaired",
        ]
    )

    summary = (
        df.groupby(["defect_type", "experiment_label"])
        .agg(
            observations=("seed", "count"),
            seeds=("seed", "nunique"),
            mean_defected=("relative_deviation_pct_defected", "mean"),
            std_defected=("relative_deviation_pct_defected", "std"),
            median_defected=("relative_deviation_pct_defected", "median"),
            mean_repaired=("relative_deviation_pct_repaired", "mean"),
            std_repaired=("relative_deviation_pct_repaired", "std"),
            median_repaired=("relative_deviation_pct_repaired", "median"),
            mean_improvement=("deviation_reduction_pct", "mean"),
            median_improvement=("deviation_reduction_pct", "median"),
            improved_cases=("repair_improved", "sum"),
            worsened_cases=("repair_worsened", "sum"),
        )
        .reset_index()
    )

    summary.to_csv(OUTPUT, index=False)

    print("\nAI DATAAUDIT — STATISTICAL ANALYSIS")
    print("=" * 75)
    print(summary.round(4).to_string(index=False))
    print(f"\nSaved: {OUTPUT}")
    print("\nNote: These are descriptive statistics.")
    print("Claims from the same seed are not independent observations.")


if __name__ == "__main__":
    main()
