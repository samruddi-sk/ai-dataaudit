import json
from pathlib import Path

import pandas as pd

from src.gemini_extractor import ExtractedClaim
from src.verify_user_claim import convert_to_claim, run_analysis


TEST_CLAIMS = [
    {
        "question": "Was Germany's revenue greater than 500000?",
        "claim": {
            "claim_type": "country_revenue_threshold",
            "country": "Germany",
            "metric": "revenue",
            "operator": ">",
            "threshold": 500000,
        },
    },
    {
        "question": "What was the total revenue in December 2010?",
        "claim": {
            "claim_type": "monthly_revenue",
            "metric": "revenue",
            "period": "2010-12",
        },
    },
    {
        "question": "Did any month generate revenue above 1000000?",
        "claim": {
            "claim_type": "monthly_revenue_threshold",
            "metric": "revenue",
            "operator": ">",
            "threshold": 1000000,
        },
    },
    {
        "question": "Which country generated the highest revenue?",
        "claim": {
            "claim_type": "top_country_revenue",
            "metric": "revenue",
        },
    },
    {
        "question": "What was the average unit price?",
        "claim": {
            "claim_type": "average_unit_price",
            "metric": "unit_price",
        },
    },
    {
        "question": "How many transaction records are there?",
        "claim": {
            "claim_type": "row_count",
            "metric": "row_count",
        },
    },
]


def main():
    print("=" * 65)
    print("          VERITYAI LOCAL BATCH TESTING")
    print("=" * 65)

    df = pd.read_parquet(
        "data/processed/reference_data.parquet"
    )

    print(f"\nDataset loaded: {len(df):,} records")
    print("Mode: LOCAL - No Gemini API calls\n")

    results = []

    for index, item in enumerate(TEST_CLAIMS, start=1):
        print(f"[{index}/{len(TEST_CLAIMS)}] {item['question']}")

        try:
            extracted = ExtractedClaim(**item["claim"])
            claim = convert_to_claim(extracted)
            result = run_analysis(df, claim)

            results.append({
                "question": item["question"],
                "claim_type": extracted.claim_type,
                "extracted_claim": extracted.model_dump(),
                "result": result,
                "status": "SUCCESS",
            })

            print(f"    Claim: {extracted.claim_type}")
            print(f"    Value: {result.get('observed_value')}")
            print("    Status: SUCCESS\n")

        except Exception as error:
            results.append({
                "question": item["question"],
                "status": "ERROR",
                "error": str(error),
            })

            print(f"    ERROR: {error}\n")

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    with open(
        output_dir / "local_batch_results.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=2, default=str)

    rows = []

    for item in results:
        result = item.get("result", {})

        rows.append({
            "question": item["question"],
            "claim_type": item.get("claim_type"),
            "status": item["status"],
            "observed_value": json.dumps(
                result.get("observed_value"),
                default=str,
            ),
            "verdict": result.get("verdict"),
            "evidence": result.get("evidence"),
            "error": item.get("error"),
        })

    pd.DataFrame(rows).to_csv(
        output_dir / "local_batch_results.csv",
        index=False,
    )

    successful = sum(
        item["status"] == "SUCCESS"
        for item in results
    )

    print("=" * 65)
    print("                 TEST SUMMARY")
    print("=" * 65)
    print(f"Total tests: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")
    print("\nSaved:")
    print("results/local_batch_results.json")
    print("results/local_batch_results.csv")


if __name__ == "__main__":
    main()
