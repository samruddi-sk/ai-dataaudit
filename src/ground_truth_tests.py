from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.claims import Claim
from src.verifier import evaluate_claim, verify_pair


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

VERDICTS = [
    "supported",
    "refuted",
    "insufficient_evidence",
    "unreliable_due_to_data_defects",
]


def make_dataset():
    """Create deterministic transaction data with four countries."""

    countries = {
        "Germany": 10.0,
        "France": 5.0,
        "Italy": 20.0,
        "Spain": 15.0,
    }

    rows = []

    for country, amount in countries.items():
        for index in range(10):
            rows.append({
                "invoice_no": f"{country[:2]}{index:03}",
                "country": country,
                "invoice_month": "2020-01",
                "line_value": amount,
                "unit_price": amount,
                "customer_id": index + 1,
            })

    return pd.DataFrame(rows)


def create_claim(
    claim_id,
    country,
    threshold,
):
    return Claim(
        claim_id=claim_id,
        text=f"{country} revenue exceeds {threshold}.",
        claim_type="country_revenue_threshold",
        country=country,
        threshold=float(threshold),
        operator=">",
        tolerance=0.05,
    )


def evaluate_direct(df, claim):
    truth, value, evidence = evaluate_claim(df, claim)

    if truth is True:
        verdict = "supported"
    elif truth is False:
        verdict = "refuted"
    else:
        verdict = "insufficient_evidence"

    return verdict, value, evidence


def build_test_cases(reference):

    cases = []

    totals = {
        "Germany": 100.0,
        "France": 50.0,
        "Italy": 200.0,
        "Spain": 150.0,
    }

    # -------------------------------------------------
    # 12 SUPPORTED CLAIMS
    # -------------------------------------------------

    supported_thresholds = [
        ("Germany", 50),
        ("Germany", 75),
        ("Germany", 99),
        ("France", 20),
        ("France", 40),
        ("France", 49),
        ("Italy", 100),
        ("Italy", 150),
        ("Italy", 199),
        ("Spain", 50),
        ("Spain", 100),
        ("Spain", 149),
    ]

    for index, (country, threshold) in enumerate(
        supported_thresholds, 1
    ):
        claim = create_claim(
            f"supported_{index:03}",
            country,
            threshold,
        )

        actual, value, evidence = evaluate_direct(
            reference, claim
        )

        cases.append({
            "test_case": f"supported_{index:03}",
            "expected_verdict": "supported",
            "actual_verdict": actual,
            "observed_value": value,
            "evidence": evidence,
        })

    # -------------------------------------------------
    # 12 REFUTED CLAIMS
    # -------------------------------------------------

    refuted_thresholds = [
        ("Germany", 101),
        ("Germany", 125),
        ("Germany", 200),
        ("France", 51),
        ("France", 75),
        ("France", 100),
        ("Italy", 201),
        ("Italy", 250),
        ("Italy", 300),
        ("Spain", 151),
        ("Spain", 175),
        ("Spain", 250),
    ]

    for index, (country, threshold) in enumerate(
        refuted_thresholds, 1
    ):
        claim = create_claim(
            f"refuted_{index:03}",
            country,
            threshold,
        )

        actual, value, evidence = evaluate_direct(
            reference, claim
        )

        cases.append({
            "test_case": f"refuted_{index:03}",
            "expected_verdict": "refuted",
            "actual_verdict": actual,
            "observed_value": value,
            "evidence": evidence,
        })

    # -------------------------------------------------
    # 8 INSUFFICIENT-EVIDENCE CLAIMS
    # -------------------------------------------------

    for index in range(8):
        country = list(totals.keys())[index % 4]

        small_dataset = reference.loc[
            reference["country"] == country
        ].head(4)

        claim = create_claim(
            f"insufficient_{index + 1:03}",
            country,
            10,
        )

        actual, value, evidence = evaluate_direct(
            small_dataset, claim
        )

        cases.append({
            "test_case": f"insufficient_{index + 1:03}",
            "expected_verdict": "insufficient_evidence",
            "actual_verdict": actual,
            "observed_value": value,
            "evidence": evidence,
        })

    # -------------------------------------------------
    # 8 DEFECT-SENSITIVE CLAIMS
    # -------------------------------------------------

    for index in range(8):
        country = list(totals.keys())[index % 4]

        original = reference.loc[
            reference["country"] == country
        ].copy()

        # Duplicate one transaction worth 10% of the
        # original country revenue.
        duplicated = pd.concat(
            [
                original,
                original.iloc[[0]],
            ],
            ignore_index=True,
        )

        claim = create_claim(
            f"defect_sensitive_{index + 1:03}",
            country,
            totals[country] * 0.5,
        )

        result = verify_pair(
            original,
            duplicated,
            claim,
        )

        cases.append({
            "test_case": f"defect_sensitive_{index + 1:03}",
            "expected_verdict": (
                "unreliable_due_to_data_defects"
            ),
            "actual_verdict": result.verdict,
            "observed_value": result.observed_value,
            "evidence": result.evidence,
            "notes": result.notes,
        })

    return cases


def calculate_metrics(cases):

    matrix = {
        expected: {
            actual: 0 for actual in VERDICTS
        }
        for expected in VERDICTS
    }

    for case in cases:
        expected = case["expected_verdict"]
        actual = case["actual_verdict"]

        if actual not in VERDICTS:
            continue

        matrix[expected][actual] += 1

    metrics = {}

    for verdict in VERDICTS:

        true_positive = matrix[verdict][verdict]

        false_positive = sum(
            matrix[other][verdict]
            for other in VERDICTS
            if other != verdict
        )

        false_negative = sum(
            matrix[verdict][other]
            for other in VERDICTS
            if other != verdict
        )

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive else 0.0
        )

        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        metrics[verdict] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
        }

    total = len(cases)

    correct = sum(
        case["expected_verdict"] == case["actual_verdict"]
        for case in cases
    )

    # False-supported rate:
    # Fraction of non-supported ground-truth cases
    # incorrectly classified as supported.
    non_supported = [
        case for case in cases
        if case["expected_verdict"] != "supported"
    ]

    false_supported = sum(
        case["actual_verdict"] == "supported"
        for case in non_supported
    )

    false_supported_rate = (
        false_supported / len(non_supported)
        if non_supported else 0.0
    )

    macro_precision = sum(
        metrics[v]["precision"] for v in VERDICTS
    ) / len(VERDICTS)

    macro_recall = sum(
        metrics[v]["recall"] for v in VERDICTS
    ) / len(VERDICTS)

    macro_f1 = sum(
        metrics[v]["f1_score"] for v in VERDICTS
    ) / len(VERDICTS)

    return {
        "confusion_matrix": matrix,
        "per_class_metrics": metrics,
        "accuracy": correct / total if total else 0.0,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "false_supported_rate": false_supported_rate,
        "correct_predictions": correct,
        "total_cases": total,
    }


def main():

    print("=" * 68)
    print("VERITYAI — EXPANDED GROUND-TRUTH EVALUATION")
    print("=" * 68)

    reference = make_dataset()

    cases = build_test_cases(reference)

    metrics = calculate_metrics(cases)

    passed = sum(
        case["expected_verdict"] == case["actual_verdict"]
        for case in cases
    )

    report = {
        "project": "VerityAI",
        "evaluation_type": (
            "constructed_controlled_ground_truth_evaluation"
        ),
        "dataset": {
            "type": "synthetic",
            "rows": len(reference),
            "countries": int(reference["country"].nunique()),
        },
        "summary": metrics,
        "cases": cases,
        "limitations": [
            "Synthetic deterministic dataset.",
            "Not an independent real-world benchmark.",
            "Tests structured claims, not Gemini extraction accuracy.",
            "Only country revenue threshold claims are evaluated.",
            "Duplicate-row defects are tested; other defect types "
            "require separate experiments.",
            "Metrics describe this constructed test set only.",
        ],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = RESULTS_DIR / "expanded_ground_truth_results.json"
    csv_path = RESULTS_DIR / "expanded_ground_truth_results.csv"

    json_path.write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )

    pd.DataFrame(cases).to_csv(csv_path, index=False)

    print(f"\nTotal test cases: {len(cases)}")
    print(f"Correct predictions: {passed}")
    print(f"Incorrect predictions: {len(cases) - passed}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")

    print("\nPER-CLASS METRICS")

    for verdict, values in metrics["per_class_metrics"].items():
        print(
            f"{verdict:35} "
            f"Precision={values['precision']:.2%} "
            f"Recall={values['recall']:.2%} "
            f"F1={values['f1_score']:.2%}"
        )

    print(
        "\nFalse-supported rate: "
        f"{metrics['false_supported_rate']:.2%}"
    )

    print("\nCONFUSION MATRIX")
    print("Rows = expected; columns = actual")

    print(
        f"{'Expected':35}",
        " | ".join(v[:12] for v in VERDICTS),
    )

    for expected in VERDICTS:
        values = [
            metrics["confusion_matrix"][expected][actual]
            for actual in VERDICTS
        ]

        print(f"{expected:35}", values)

    print("\nSaved:")
    print(json_path)
    print(csv_path)

    print(
        "\nThis is a controlled synthetic evaluation, "
        "not an independent scientific benchmark."
    )


if __name__ == "__main__":
    main()
