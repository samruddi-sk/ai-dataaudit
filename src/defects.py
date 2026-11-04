from __future__ import annotations

import numpy as np
import pandas as pd


DEFECT_ALIASES = {
    "duplicates": "duplicate_rows",
    "duplicate_rows": "duplicate_rows",
    "cancellations": "drop_cancellations",
    "drop_cancellations": "drop_cancellations",
    "missing_customer_ids": "missing_customer_ids",
    "invalid_values": "invalid_prices",
    "invalid_prices": "invalid_prices",
    "missing_transactions": "missing_transactions",
}


REPAIR_ALIASES = {
    "duplicates": "dedupe",
    "duplicate_rows": "dedupe",
    "dedupe": "dedupe",
    "cancellations": "drop_cancellations",
    "drop_cancellations": "drop_cancellations",
    "invalid_values": "filter_invalid_prices_quantities",
    "invalid_prices": "filter_invalid_prices_quantities",
    "filter_invalid_prices_quantities": "filter_invalid_prices_quantities",
    "missing_customer_ids": "preserve_missing_customer_ids",
    "preserve_missing_customer_ids": "preserve_missing_customer_ids",
    "drop_missing_customer_ids": "drop_missing_customer_ids",
    "missing_transactions": "no_repair",
    "no_repair": "no_repair",
}


def inject_defect(
    df: pd.DataFrame,
    defect_type: str,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Inject a reproducible defect and return the dataset and metadata."""

    if df.empty:
        raise ValueError("Cannot inject defects into an empty dataframe.")

    if defect_type not in DEFECT_ALIASES:
        raise ValueError(
            f"Unknown defect type: {defect_type}. "
            f"Supported types: {sorted(DEFECT_ALIASES)}"
        )

    defect_type = DEFECT_ALIASES[defect_type]
    rng = np.random.default_rng(seed)
    out = df.copy(deep=True)

    metadata = {
        "defect_type": defect_type,
        "seed": int(seed),
        "rows_before": int(len(out)),
    }

    if defect_type == "duplicate_rows":
        count = max(1, int(len(out) * 0.01))

        chosen = out.sample(
            n=min(count, len(out)),
            random_state=seed,
        )

        out = pd.concat([out, chosen], ignore_index=True)

        metadata["rows_injected"] = int(len(chosen))

    elif defect_type == "drop_cancellations":
        count = max(1, int(len(out) * 0.005))

        chosen = out.sample(
            n=min(count, len(out)),
            random_state=seed,
        ).copy()

        chosen["invoice_no"] = (
            "C" + chosen["invoice_no"].astype(str).str.lstrip("C")
        )

        chosen["line_value"] = (
            chosen["quantity"] * chosen["unit_price"]
        )

        out = pd.concat([out, chosen], ignore_index=True)

        metadata["rows_injected"] = int(len(chosen))
        metadata["note"] = (
            "Synthetic cancellation-like rows were added. "
            "These are not recovered historical cancellation records."
        )

    elif defect_type == "missing_customer_ids":
        if "customer_id" not in out.columns:
            raise ValueError(
                "customer_id column is required for this defect."
            )

        count = max(1, int(len(out) * 0.05))

        indices = rng.choice(
            out.index.to_numpy(),
            size=min(count, len(out)),
            replace=False,
        )

        out.loc[indices, "customer_id"] = pd.NA

        metadata["rows_affected"] = int(len(indices))
        metadata["note"] = (
            "Customer identifiers were removed; transaction values "
            "were deliberately preserved."
        )

    elif defect_type == "invalid_prices":
        count = max(1, int(len(out) * 0.005))

        indices = rng.choice(
            out.index.to_numpy(),
            size=min(count, len(out)),
            replace=False,
        )

        out.loc[indices, "unit_price"] = -1.0

        if "line_value" in out.columns:
            out.loc[indices, "line_value"] = (
                out.loc[indices, "quantity"] * -1.0
            )

        metadata["rows_affected"] = int(len(indices))

    elif defect_type == "missing_transactions":
        count = max(1, int(len(out) * 0.02))

        indices = rng.choice(
            out.index.to_numpy(),
            size=min(count, len(out)),
            replace=False,
        )

        removed = out.loc[indices].copy()

        # Calculate the information loss before deleting the records.
        if "line_value" in out.columns:
            total_revenue = pd.to_numeric(
                out["line_value"],
                errors="coerce",
            ).sum()

            removed_revenue = pd.to_numeric(
                removed["line_value"],
                errors="coerce",
            ).sum()

            metadata["total_reference_revenue"] = float(total_revenue)
            metadata["revenue_removed"] = float(removed_revenue)

            if total_revenue != 0:
                metadata["revenue_removed_pct"] = float(
                    abs(removed_revenue / total_revenue) * 100
                )
            else:
                metadata["revenue_removed_pct"] = None

        if "quantity" in out.columns:
            metadata["quantity_removed"] = float(
                pd.to_numeric(
                    removed["quantity"],
                    errors="coerce",
                ).sum()
            )

        metadata["rows_removed"] = int(len(removed))
        metadata["rows_removed_pct"] = float(
            len(removed) / len(out) * 100
        )

        out = out.drop(index=indices).reset_index(drop=True)

        metadata["repair_operator"] = None
        metadata["note"] = (
            "Transactions were removed. Their original values cannot "
            "be recovered from the remaining records alone."
        )

    metadata["rows_after"] = int(len(out))

    return out.reset_index(drop=True), metadata


def apply_repair(
    df: pd.DataFrame,
    repair_type: str,
) -> tuple[pd.DataFrame, dict]:
    """Apply a repair strategy and return the data and metadata."""

    if repair_type not in REPAIR_ALIASES:
        raise ValueError(
            f"Unknown repair type: {repair_type}. "
            f"Supported types: {sorted(REPAIR_ALIASES)}"
        )

    normalized_repair = REPAIR_ALIASES[repair_type]
    out = df.copy(deep=True)

    metadata = {
        "repair_type": normalized_repair,
        "rows_before": int(len(out)),
    }

    if normalized_repair == "dedupe":
        out = out.drop_duplicates().reset_index(drop=True)

    elif normalized_repair == "drop_cancellations":
        if "invoice_no" not in out.columns:
            raise ValueError(
                "invoice_no column is required to remove cancellations."
            )

        mask = (
            out["invoice_no"]
            .fillna("")
            .astype(str)
            .str.upper()
            .str.startswith("C")
        )

        out = out.loc[~mask].copy().reset_index(drop=True)

    elif normalized_repair == "filter_invalid_prices_quantities":
        required = {"unit_price", "quantity"}
        missing = required - set(out.columns)

        if missing:
            raise ValueError(
                f"Missing columns required for validation: {sorted(missing)}"
            )

        prices = pd.to_numeric(out["unit_price"], errors="coerce")
        quantities = pd.to_numeric(out["quantity"], errors="coerce")

        valid = (
            prices.notna()
            & quantities.notna()
            & (prices > 0)
            & (quantities > 0)
        )

        out = out.loc[valid].copy().reset_index(drop=True)

        metadata["rows_removed"] = int(
            metadata["rows_before"] - len(out)
        )

        metadata["note"] = (
            "Rows with nonpositive or nonnumeric prices or quantities "
            "were removed. Validate this policy against the dataset "
            "semantics before treating it as ground truth."
        )

    elif normalized_repair == "preserve_missing_customer_ids":
        metadata["rows_changed"] = 0
        metadata["rows_removed"] = 0

        metadata["note"] = (
            "Conservative strategy: preserve transactions with missing "
            "customer IDs. No customer identity was imputed."
        )

    elif normalized_repair == "drop_missing_customer_ids":
        if "customer_id" not in out.columns:
            raise ValueError(
                "customer_id column is required for this repair."
            )

        out = (
            out.loc[out["customer_id"].notna()]
            .copy()
            .reset_index(drop=True)
        )

        metadata["rows_removed"] = int(
            metadata["rows_before"] - len(out)
        )

        metadata["note"] = (
            "Aggressive sensitivity strategy: remove records without "
            "customer IDs. This can introduce selection bias and is "
            "not assumed to be a correct repair."
        )

    elif normalized_repair == "no_repair":
        metadata["note"] = (
            "No repair applied. Missing transactions cannot be "
            "reconstructed reliably from the remaining records alone."
        )

        metadata["rows_recovered"] = 0

    metadata["rows_after"] = int(len(out))

    return out, metadata
