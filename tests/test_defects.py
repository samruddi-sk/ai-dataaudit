import pandas as pd
from src.defects import inject_defect, apply_repair


def sample_df():
    return pd.DataFrame({
        "invoice_no": ["100", "101", "102", "103"],
        "stock_code": ["A", "B", "C", "D"],
        "quantity": [1, 2, 1, 3],
        "unit_price": [10.0, 5.0, 3.0, 2.0],
        "line_revenue": [10.0, 10.0, 3.0, 6.0],
        "customer_id": ["1", "2", "3", "4"],
        "country": ["A", "A", "B", "B"],
        "invoice_date": pd.to_datetime(["2020-01-01"] * 4),
        "invoice_month": ["2020-01"] * 4,
    })


def test_duplicate_injection_and_repair():
    df = sample_df()
    defected, _ = inject_defect(df, "duplicate_rows", seed=3)
    assert len(defected) > len(df)
    repaired, _ = apply_repair(defected, "dedupe")
    assert len(repaired) == len(df)


def test_missing_transactions_is_no_repair():
    df = sample_df()
    defected, metadata = inject_defect(df, "missing_transactions", seed=1)
    repaired, info = apply_repair(defected, "no_repair")
    assert len(repaired) < len(df)
    assert info["note"].startswith("No repair")
    assert metadata["repair_operator"] is None


def test_input_not_mutated():
    df = sample_df()
    original = df.copy(deep=True)
    inject_defect(df, "invalid_prices", seed=1)
    pd.testing.assert_frame_equal(df, original)
