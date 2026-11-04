from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.run_benchmark import (
    EXPERIMENTS,
    RESULTS_DIR,
    claim_text,
    evaluate_dataset,
    load_reference,
)
from src.claims import make_default_claims
from src.defects import inject_defect, apply_repair


SEEDS = [42, 123, 2026]

OUTPUT_FILE = RESULTS_DIR / "seed_experiment_results.csv"
SUMMARY_FILE = RESULTS_DIR / "seed_experiment_summary.json"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("VERITYAI - MULTI-SEED REPRODUCIBILITY EXPERIMENT")
    print("=" * 68)

    reference = load_reference()
    claims = make_default_claims(reference)

    print(f"Reference records: {len(reference):,}")
    print(f"Claims: {len(claims)}")
    print(f"Seeds: {SEEDS}")

    rows: list[dict[str, Any]] = []

    for seed in SEEDS:
        print(f"\nSEED {seed}")

        for defect_type, repair_type, label in EXPERIMENTS:
            defective = None
            repaired = None

            print(f"  Experiment: {label}")

            try:
                defective, defect_metadata = inject_defect(
                    reference,
                    defect_type,
                    seed=seed,
                )

                defect_rows = evaluate_dataset(
                    reference=reference,
                    dataset=defective,
                    claims=claims,
                    version="defected",
                    defect=defect_type,
                    repair="none",
                    experiment_label=label,
                    defect_metadata=defect_metadata,
                )

                for row in defect_rows:
                    row["seed"] = seed
                rows.extend(defect_rows)

                repaired, repair_metadata = apply_repair(
                    defective,
                    repair_type,
                )

                repaired_rows = evaluate_dataset(
                    reference=reference,
                    dataset=repaired,
                    claims=claims,
                    version="repaired",
                    defect=defect_type,
                    repair=repair_type,
                    experiment_label=label,
                    defect_metadata=defect_metadata,
                    repair_metadata=repair_metadata,
                )

                for row in repaired_rows:
                    row["seed"] = seed
                rows.extend(repaired_rows)

            except Exception as exc:
                print(
                    f"    Experiment failed: "
                    f"{type(exc).__name__}: {exc}"
                )

                rows.append(
                    {
                        "seed": seed,
                        "claim_id": 0,
                        "claim": "Experiment pipeline",
                        "dataset_version": "error",
                        "defect_type": defect_type,
                        "experiment_label": label,
                        "repair_applied": repair_type,
                        "verdict": "experiment_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

            finally:
                del defective
                del repaired
                gc.collect()

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_FILE, index=False)

    valid = results[
        results["dataset_version"].isin(["defected", "repaired"])
    ].copy()

    summary = {
        "project": "VerityAI",
        "seeds": SEEDS,
        "claims_per_seed": len(claims),
        "experiment_conditions": len(EXPERIMENTS),
        "result_rows": len(results),
        "expected_result_rows": (
            len(SEEDS) * len(EXPERIMENTS) * len(claims) * 2
        ),
        "experiment_errors": int(
            (results["verdict"] == "experiment_error").sum()
        ),
        "verdict_counts": {
            str(key): int(value)
            for key, value in results["verdict"].value_counts().items()
        },
        "results_file": str(OUTPUT_FILE),
        "note": (
            "This experiment evaluates sensitivity to random defect "
            "selection. Claims are still generated from the reference "
            "dataset, so this is not an independent accuracy evaluation."
        ),
    }

    SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 68)
    print("MULTI-SEED EXPERIMENT FINISHED")
    print("=" * 68)
    print(json.dumps(summary, indent=2))
    print(f"\nDetailed results: {OUTPUT_FILE}")
    print(f"Summary: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
