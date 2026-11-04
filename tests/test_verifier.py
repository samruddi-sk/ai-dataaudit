import pandas as pd
from src.claims import Claim
from src.verifier import evaluate_claim, verify_pair


def sample_df():
    return pd.DataFrame({
        "invoice_no": ["1", "2", "3", "4", "5", "6"],
        "stock_code": ["A", "B", "C", "D", "E", "F"],
        "quantity": [1] * 6,
        "unit_price": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "line_revenue": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "customer_id": ["1", "2", "3", "4", "5", "6"],
        "country": ["Germany"] * 6,
        "invoice_date": pd.to_datetime(["2020-01-01"] * 6),
        "invoice_month": ["2020-01"] * 6,
    })


def test_country_claim_supported():
    df = sample_df()
    claim = Claim(
        claim_id="t1",
        template_id="country_revenue_threshold",
        metric="line_revenue",
        country="Germany",
        threshold=100,
        operator=">",
        text="test",
    )
    truth, value, evidence = evaluate_claim(df, claim)
    assert truth is True
    assert value == 210.0


def test_reference_indeterminate_has_precedence():
    df = sample_df()
    claim = Claim(
        claim_id="t2",
        template_id="country_revenue_threshold",
        metric="line_revenue",
        country="Unknown",
        threshold=100,
        operator=">",
        text="test",
    )
    result = verify_pair(df, df, claim)
    assert result.verdict == "insufficient_evidence"
