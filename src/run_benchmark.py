from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from src.claims import make_default_claims
from src.defects import inject_defect, apply_repair
from src.verifier import verify_pair


ROOT = Path(__file__).resolve().parents[1]
DB_FILE = ROOT / "data" / "processed" / "verityai.duckdb"
RESULTS_DIR = ROOT / "results"
RESULTS_FILE = RESULTS_DIR / "benchmark_results.csv"
SUMMARY_FILE = RESULTS_DIR / "benchmark_summary.json"


# Each tuple contains:
# (defect name, repair strategy, experiment label).
EXPERIMENTS = [
    (
        "duplicates",
        "dedupe",
        "duplicate_rows",
    ),
    (
        "cancellations",
        "drop_cancellations",
        "synthetic_cancellations",
    ),
    (
        "missing_customer_ids",
        "preserve_missing_customer_ids",
        "missing_ids_preserved",
    ),
    (
        "missing_customer_ids",
        "drop_missing_customer_ids",
        "missing_ids_removed",
    ),
    (
        "invalid_values",
        "filter_invalid_prices_quantities",
        "invalid_prices",
    ),
    (
        "missing_transactions",
        "no_repair",
        "missing_transactions_unrecoverable",
    ),
]


def load_reference() -> pd.DataFrame:
    """Load the reference dataset from DuckDB."""
    if not DB_FILE.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_FILE}\n"
            "Run: python3 -m src.prepare_data"
        )

    with duckdb.connect(str(DB_FILE), read_only=True) as connection:
        tables = connection.execute("SHOW TABLES").fetchdf()["name"].tolist()

        if "reference_data" not in tables:
            raise RuntimeError(
                f"Missing reference_data table. Tables found: {tables}"
            )

        return connection.execute(
            "SELECT * FROM reference_data"
        ).fetchdf()


def claim_text(claim: Any) -> str:
    """Return a readable claim."""
    return str(getattr(claim, "text", claim))


def get_field(result: Any, name: str) -> Any:
    """Read a field from a result object or dictionary."""
    if result is None:
        return None

    if isinstance(result, dict):
        return result.get(name)

    return getattr(result, name, None)


def get_verdict(result: Any) -> str:
    """Extract the verdict from a verification result."""
    verdict = get_field(result, "verdict")

    if verdict is None:
        return "unknown"

    return str(getattr(verdict, "value", verdict))


def make_row(
    claim_id: int,
    claim: Any,
    version: str,
    defect: str,
    repair: str,
    experiment_label: str,
    result: Any = None,
    error: str = "",
    defect_metadata: dict[str, Any] | None = None,
    repair_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Format one benchmark result."""
    return {
        "claim_id": claim_id,
        "claim": claim_text(claim),
        "dataset_version": version,
        "defect_type": defect,
        "experiment_label": experiment_label,
        "repair_applied": repair,
        "verdict": get_verdict(result),
        "observed_value": get_field(result, "observed_value"),
        "reference_value": get_field(result, "reference_value"),
        "confidence": get_field(result, "confidence"),
        "repair_sensitivity": get_field(result, "repair_sensitivity"),
        "evidence": json.dumps(
            get_field(result, "evidence"),
            default=str,
            ensure_ascii=False,
        ),
        "defect_metadata": json.dumps(
            defect_metadata or {},
            default=str,
            ensure_ascii=False,
        ),
        "repair_metadata": json.dumps(
            repair_metadata or {},
            default=str,
            ensure_ascii=False,
        ),
        "error": error,
    }


def evaluate_dataset(
    reference: pd.DataFrame,
    dataset: pd.DataFrame,
    claims: list[Any],
    version: str,
    defect: str,
    repair: str,
    experiment_label: str,
    defect_metadata: dict[str, Any] | None = None,
    repair_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate every claim on one dataset version."""
    rows = []

    for index, claim in enumerate(claims, start=1):
        try:
            result = verify_pair(reference, dataset, claim)

            rows.append(
                make_row(
                    index,
                    claim,
                    version,
                    defect,
                    repair,
                    experiment_label,
                    result=result,
                    defect_metadata=defect_metadata,
                    repair_metadata=repair_metadata,
                )
            )

        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"

            print(
                f"  Claim {index} failed on "
                f"{experiment_label}/{version}: {message}"
            )

            rows.append(
                make_row(
                    index,
                    claim,
                    version,
                    defect,
                    repair,
                    experiment_label,
                    error=message,
                    defect_metadata=defect_metadata,
                    repair_metadata=repair_metadata,
                )
            )

    return rows


def main() -> None:
    print("=" * 68)
    print("AI DATAAUDIT - REPAIR-ROBUST CLAIM VERIFICATION BENCHMARK")
    print("=" * 68)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    reference = load_reference()
    print(f"Loaded {len(reference):,} reference records.")

    claims = make_default_claims(reference)

    if not claims:
        raise RuntimeError(
            "No claims generated. Check src/claims.py."
        )

    print(f"Generated {len(claims)} claims.")
    results = []

    print("\nEvaluating reference baseline...")

    results.extend(
        evaluate_dataset(
            reference=reference,
            dataset=reference,
            claims=claims,
            version="reference",
            defect="none",
            repair="none",
            experiment_label="baseline",
        )
    )

    for index, (defect_type, repair_type, label) in enumerate(
        EXPERIMENTS,
        start=1,
    ):
        print(
            f"\nExperiment {index}/{len(EXPERIMENTS)}: "
            f"{label} ({defect_type} -> {repair_type})"
        )

        defective = None
        repaired = None

        try:
            defective, defect_metadata = inject_defect(
                reference,
                defect_type,
                seed=42,
            )

            print(f"  Defected rows: {len(defective):,}")

            results.extend(
                evaluate_dataset(
                    reference=reference,
                    dataset=defective,
                    claims=claims,
                    version="defected",
                    defect=defect_type,
                    repair="none",
                    experiment_label=label,
                    defect_metadata=defect_metadata,
                )
            )

            repaired, repair_metadata = apply_repair(
                defective,
                repair_type,
            )

            print(f"  Post-repair rows: {len(repaired):,}")

            results.extend(
                evaluate_dataset(
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
            )

            if repair_type == "no_repair":
                print(
                    "  NOTE: No reconstruction attempted. "
                    "Deleted transactions remain unrecoverable."
                )

        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            print(f"  Experiment failed: {message}")

            results.append(
                make_row(
                    claim_id=0,
                    claim="Defect/repair pipeline",
                    version="error",
                    defect=defect_type,
                    repair=repair_type,
                    experiment_label=label,
                    error=message,
                )
            )

        finally:
            del defective
            del repaired
            gc.collect()

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_FILE, index=False)

    experiment_errors = int(
        results_df["error"].fillna("").astype(str).ne("").sum()
    )

    summary = {
        "project": "AI DataAudit",
        "dataset": "UCI Online Retail II",
        "reference_rows": int(len(reference)),
        "claims_generated": int(len(claims)),
        "number_of_experiment_conditions": len(EXPERIMENTS),
        "experiment_conditions": [
            {
                "defect_type": defect,
                "repair_type": repair,
                "label": label,
            }
            for defect, repair, label in EXPERIMENTS
        ],
        "experiment_rows": int(len(results_df)),
        "verdict_counts": {
            str(key): int(value)
            for key, value in (
                results_df["verdict"].value_counts().items()
            )
        },
        "experiment_errors": experiment_errors,
        "results_file": str(RESULTS_FILE),
        "interpretation_note": (
            "Prototype diagnostic benchmark. Claims are generated from "
            "the reference dataset. Repair strategies are experimental; "
            "matching reference metrics does not prove a repair is valid. "
            "Missing transactions are intentionally left unrepaired because "
            "deleted records cannot be reconstructed from the remaining "
            "dataset alone."
        ),
    }

    SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 68)
    print("BENCHMARK FINISHED")
    print("=" * 68)
    print(json.dumps(summary, indent=2))
    print(f"\nDetailed results: {RESULTS_FILE}")
    print(f"Summary: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
