from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def save_chart(name):
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=300, bbox_inches="tight")
    plt.close()


stats = pd.read_csv(RESULTS / "statistical_analysis.csv")
robustness = pd.read_csv(RESULTS / "defect_robustness_results.csv")
results = pd.read_csv(RESULTS / "seed_experiment_results.csv")


# 1. Deviation before and after repair
plt.figure(figsize=(13, 6))
x = range(len(stats))
width = 0.35

plt.bar(
    [i - width / 2 for i in x],
    stats["mean_defected"],
    width,
    label="Defected",
)
plt.bar(
    [i + width / 2 for i in x],
    stats["mean_repaired"],
    width,
    label="Repaired",
)
plt.xticks(list(x), stats["experiment_label"], rotation=25, ha="right")
plt.ylabel("Mean deviation (%)")
plt.title("Metric Deviation Before and After Repair")
plt.legend()
save_chart("01_deviation_comparison.png")


# 2. Deviation variability
plt.figure(figsize=(13, 6))
plt.bar(stats["experiment_label"], stats["std_defected"])
plt.xticks(rotation=25, ha="right")
plt.ylabel("Standard deviation (%)")
plt.title("Variation in Defected Dataset Metrics")
save_chart("02_deviation_variability.png")


# 3. Repair effectiveness
plt.figure(figsize=(13, 6))
plt.bar(stats["experiment_label"], stats["mean_improvement"])
plt.axhline(0, linewidth=1)
plt.xticks(rotation=25, ha="right")
plt.ylabel("Mean deviation reduction (percentage points)")
plt.title("Effect of Repair Strategies")
save_chart("03_repair_effectiveness.png")


# 4. Missing transaction revenue loss
loss_rows = []

missing = results[
    (results["defect_type"] == "missing_transactions")
    & (results["dataset_version"] == "defected")
]

for row in missing.itertuples():
    try:
        metadata = json.loads(row.defect_metadata)
        value = metadata.get("revenue_removed_pct")

        if value is not None:
            loss_rows.append({
                "seed": row.seed,
                "revenue_loss_pct": float(value),
            })
    except (TypeError, ValueError, json.JSONDecodeError):
        continue

loss = pd.DataFrame(loss_rows).drop_duplicates("seed")

plt.figure(figsize=(8, 5))

if not loss.empty:
    plt.bar(
        loss["seed"].astype(str),
        loss["revenue_loss_pct"],
    )
    plt.ylabel("Revenue lost (%)")
    plt.xlabel("Random seed")
else:
    plt.text(0.5, 0.5, "Revenue metadata unavailable", ha="center")

plt.title("Financial Impact of Missing Transactions")
save_chart("04_missing_transaction_loss.png")


# 5. Verdict distribution
verdicts = results.groupby(
    ["dataset_version", "verdict"]
).size().unstack(fill_value=0)

verdicts.plot(kind="bar", figsize=(10, 6))
plt.ylabel("Number of evaluations")
plt.xlabel("Dataset version")
plt.title("Verification Verdict Distribution")
plt.xticks(rotation=0)
save_chart("05_verdict_distribution.png")


# 6. Repair effects across seeds
plt.figure(figsize=(13, 6))

for label, group in robustness.groupby("experiment_label"):
    plt.scatter(
        group["seed"],
        group["deviation_reduction_pct"],
        label=label,
    )

plt.axhline(0, linewidth=1)
plt.xlabel("Random seed")
plt.ylabel("Deviation reduction (%)")
plt.title("Repair Effects Across Random Seeds")
plt.legend(fontsize=8)
save_chart("06_seed_repair_effects.png")


print("\nAI DATAAUDIT FIGURES GENERATED")
print("=" * 50)

for path in sorted(FIGURES.glob("*.png")):
    print(path.name)

print(f"\nSaved in: {FIGURES}")
