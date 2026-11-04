from __future__ import annotations

import json
import os
import time
from typing import Literal, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import errors
from pydantic import BaseModel, Field

load_dotenv()

ClaimType = Literal[
    "country_revenue_threshold",
    "monthly_revenue",
    "monthly_revenue_threshold",
    "top_country_revenue",
    "average_unit_price",
    "row_count",
]


class ExtractedClaim(BaseModel):
    claim_type: ClaimType
    country: Optional[str] = None
    metric: Literal["revenue", "unit_price", "row_count"]
    period: Optional[str] = None
    operator: Optional[Literal[">", ">=", "<", "<=", "=="]] = None
    threshold: Optional[float] = None
    needs_clarification: bool = False
    clarification_question: Optional[str] = None


SYSTEM_INSTRUCTIONS = """
Extract structured claims from questions about a retail dataset.

Supported claim types:
- country_revenue_threshold
- monthly_revenue
- monthly_revenue_threshold
- top_country_revenue
- average_unit_price
- row_count

Rules:
1. Extract only explicitly stated or clearly implied information.
2. Never invent countries, dates, thresholds or results.
3. Never calculate answers.
4. Use YYYY-MM for specified months.
5. Country revenue threshold requires country, operator and threshold.
6. Monthly revenue requires a month.
7. Monthly revenue threshold requires operator and threshold.
8. Top country revenue requires no country.
9. Average unit price and row count require no threshold.
10. Ask for clarification when the question is ambiguous.
"""


def get_client():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing from .env.")

    return genai.Client(api_key=api_key)


def validate_claim(claim: ExtractedClaim) -> ExtractedClaim:
    if claim.needs_clarification:
        return claim

    if claim.claim_type == "country_revenue_threshold":
        if not claim.country or claim.operator is None or claim.threshold is None:
            raise ValueError("Country revenue claim is missing required fields.")

    elif claim.claim_type == "monthly_revenue":
        if not claim.period:
            raise ValueError("Monthly revenue claim requires a month.")

    elif claim.claim_type == "monthly_revenue_threshold":
        if claim.operator is None or claim.threshold is None:
            raise ValueError("Monthly threshold claim requires operator and threshold.")

    return claim


def extract_claim(
    text: str,
    model: Optional[str] = None,
    max_attempts: int = 3,
) -> ExtractedClaim:

    if not text or not text.strip():
        raise ValueError("Claim text cannot be empty.")

    client = get_client()
    model_name = model or os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

    prompt = f"""
{SYSTEM_INSTRUCTIONS}

User question:
{text.strip()}
"""

    for attempt in range(max_attempts):
        try:
            print(f"      Gemini request attempt {attempt + 1}/{max_attempts}")

            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": ExtractedClaim,
                    "temperature": 0,
                },
            )

            if not response.text:
                raise RuntimeError("Gemini returned an empty response.")

            data = json.loads(response.text)
            claim = ExtractedClaim.model_validate(data)

            return validate_claim(claim)

        except errors.APIError as exc:
            retryable = exc.code in {429, 500, 502, 503, 504}

            if not retryable or attempt == max_attempts - 1:
                raise

            delay = 2 ** (attempt + 1)

            print(
                f"      Gemini temporarily unavailable. "
                f"Retrying in {delay} seconds..."
            )
            time.sleep(delay)

    raise RuntimeError("Gemini request failed after all retry attempts.")


def main():
    examples = [
        "Germany revenue was greater than 500000.",
        "What was the revenue in December 2010?",
        "Did any month generate more than 1000000?",
        "Which country generated the highest revenue?",
        "What was the average unit price?",
        "How many transaction records are there?",
    ]

    for question in examples:
        print("\n" + "=" * 65)
        print("QUESTION:", question)

        try:
            result = extract_claim(question)
            print(result.model_dump_json(indent=2))
        except Exception as exc:
            print(f"ERROR: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
