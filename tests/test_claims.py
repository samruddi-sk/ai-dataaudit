import pandas as pd
from src.claims import Claim, make_default_claims, parse_claim_text


def test_parse_country_revenue_claim():
    claim = parse_claim_text("Total revenue for Germany was greater than 10000.")
    assert claim.template_id == "country_revenue_threshold"
    assert claim.country == "Germany"
    assert claim.threshold == 10000.0


def test_claim_schema_rejects_invalid_top_n():
    try:
        Claim(
            claim_id="x",
            template_id="top_products",
            metric="line_revenue",
            top_n=100,
            text="test",
        )
    except Exception:
        return
    raise AssertionError("Expected invalid top_n to fail")


def test_default_claims_are_deterministic():
    df = pd.DataFrame({
        "country": ["A", "A", "B", "B"] * 20,
        "invoice_month": ["2020-01", "2020-02", "2020-01", "2020-02"] * 20,
        "line_revenue": [10, 12, 20, 18] * 20,
        "stock_code": ["X", "Y", "X", "Z"] * 20,
    })
    a = make_default_claims(df, limit=8, seed=1)
    b = make_default_claims(df, limit=8, seed=1)
    assert [x.model_dump() for x in a] == [x.model_dump() for x in b]


def test_claim_top_n_syncs_with_parameters():
    claim = Claim(
        claim_id="test-top-n",
        claim_type="top_country_revenue",
        parameters={"top_n": 3},
    )
    assert claim.top_n == 3
    assert claim.parameters["top_n"] == 3


def test_claim_top_n_is_saved_in_parameters():
    claim = Claim(
        claim_id="test-top-n",
        claim_type="top_country_revenue",
        top_n=5,
    )
    assert claim.parameters["top_n"] == 5


def test_run_analysis_returns_top_n_countries():
    from src.verify_user_claim import run_analysis

    df = pd.DataFrame({
        "country": ["A", "B", "C", "D", "A", "B"],
        "line_revenue": [100, 300, 200, 50, 150, 100],
    })

    claim = Claim(
        claim_id="top-3-test",
        claim_type="top_country_revenue",
        top_n=3,
    )

    result = run_analysis(df, claim)

    assert [item["country"] for item in result["observed_value"]] == [
        "B", "A", "C"
    ]
    assert [item["revenue"] for item in result["observed_value"]] == [
        400.0, 250.0, 200.0
    ]
