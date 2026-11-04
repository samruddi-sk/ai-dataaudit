from __future__ import annotations
from pathlib import Path
import hashlib
import json
import duckdb
import pandas as pd
# --------------------------------------------------
# PATHS
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_FILE = RAW_DIR / "online_retail_ii.csv"
RAW_EXCEL_FILE = RAW_DIR / "online_retail_II.xlsx"
REFERENCE_FILE = PROCESSED_DIR / "reference_data.parquet"
DATABASE_FILE = PROCESSED_DIR / "verityai.duckdb"
METADATA_FILE = PROCESSED_DIR / "metadata.json"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
# --------------------------------------------------
# FILE HASHING
# --------------------------------------------------
def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 hash of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()
# --------------------------------------------------
# COLUMN NORMALIZATION
# --------------------------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Online Retail II column names."""
    df = df.copy()
    aliases = {
        "invoiceno": "invoice_no",
        "invoice": "invoice_no",
        "stockcode": "stock_code",
        "description": "description",
        "quantity": "quantity",
        "invoicedate": "invoice_date",
        "price": "unit_price",
        "unitprice": "unit_price",
        "customerid": "customer_id",
        "country": "country",
    }
    mapping = {}
    for col in df.columns:
        key = (
            str(col)
            .strip()
            .lower()
            .replace("_", "")
            .replace(" ", "")
        )
        if key in aliases:
            mapping[col] = aliases[key]
    df = df.rename(columns=mapping)
    required = {
        "invoice_no",
        "stock_code",
        "quantity",
        "invoice_date",
        "unit_price",
        "country",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Dataset columns did not match the expected schema.\n"
            f"Missing columns: {sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )
    if "customer_id" not in df.columns:
        df["customer_id"] = pd.NA
    if "description" not in df.columns:
        df["description"] = pd.NA
    return df
# --------------------------------------------------
# LOAD RAW DATASET
# --------------------------------------------------
def load_raw_dataset() -> pd.DataFrame:
    """
    Load Online Retail II from an existing CSV cache
    or the Excel workbook in data/raw/.
    """
    # Option 1: Use an existing CSV cache.
    if RAW_FILE.exists():
        print(f"Loading cached CSV: {RAW_FILE}")
        return pd.read_csv(
            RAW_FILE,
            dtype={
                "Invoice": "string",
                "InvoiceNo": "string",
                "invoice_no": "string",
                "StockCode": "string",
                "stock_code": "string",
            },
            low_memory=False,
        )
    # Option 2: Read the Excel workbook.
    if RAW_EXCEL_FILE.exists():
        print(f"Loading Excel workbook: {RAW_EXCEL_FILE}")
        print("Reading workbook sheets. This may take a while...")
        sheets = pd.read_excel(
            RAW_EXCEL_FILE,
            sheet_name=None,
            engine="openpyxl",
        )
        frames = []
        for sheet_name, sheet_df in sheets.items():
            if not sheet_df.empty:
                print(
                    f"Sheet '{sheet_name}': "
                    f"{len(sheet_df):,} rows"
                )
                frames.append(sheet_df)
        if not frames:
            raise ValueError(
                "The Excel workbook contains no data."
            )
        df = pd.concat(
            frames,
            ignore_index=True,
        )
        print(
            f"Total imported rows: {len(df):,}"
        )
        # Save a CSV cache for subsequent runs.
        print("Saving CSV cache...")
        df.to_csv(
            RAW_FILE,
            index=False,
        )
        print(f"CSV cache saved: {RAW_FILE}")
        return df
    raise FileNotFoundError(
        "Online Retail II was not found.\n"
        f"Expected Excel file: {RAW_EXCEL_FILE}\n"
        f"Expected CSV file: {RAW_FILE}\n"
        "Place the downloaded dataset in data/raw/."
    )
# --------------------------------------------------
# CREATE REFERENCE DATASET
# --------------------------------------------------
def create_reference_dataset(
    raw_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Create a documented reference dataset.
    The reference dataset applies deterministic cleaning
    rules. It is not assumed to represent perfect ground truth.
    """
    df = normalize_columns(raw_df)
    initial_rows = len(df)
    # Normalize identifiers and categorical columns.
    df["invoice_no"] = (
        df["invoice_no"]
        .astype("string")
        .str.strip()
    )
    df["stock_code"] = (
        df["stock_code"]
        .astype("string")
        .str.strip()
    )
    df["country"] = (
        df["country"]
        .astype("string")
        .str.strip()
    )
    # Convert numeric and date fields.
    df["invoice_date"] = pd.to_datetime(
        df["invoice_date"],
        errors="coerce",
    )
    df["quantity"] = pd.to_numeric(
        df["quantity"],
        errors="coerce",
    )
    df["unit_price"] = pd.to_numeric(
        df["unit_price"],
        errors="coerce",
    )
    # Normalize customer identifiers.
    df["customer_id"] = (
        df["customer_id"]
        .astype("string")
        .str.strip()
    )
    df["customer_id"] = df["customer_id"].replace(
        {
            "": pd.NA,
            "nan": pd.NA,
            "<NA>": pd.NA,
        }
    )
    # Remove exact duplicate rows.
    rows_before = len(df)
    df = df.drop_duplicates()
    duplicates_removed = rows_before - len(df)
    # Exclude cancellation invoices.
    rows_before = len(df)
    cancellation_mask = (
        df["invoice_no"]
        .fillna("")
        .str.upper()
        .str.startswith("C")
    )
    df = df.loc[~cancellation_mask].copy()
    cancellations_removed = rows_before - len(df)
    # Remove records missing essential transaction fields.
    rows_before = len(df)
    df = df.dropna(
        subset=[
            "invoice_no",
            "stock_code",
            "quantity",
            "invoice_date",
            "unit_price",
            "country",
        ]
    ).copy()
    incomplete_rows_removed = rows_before - len(df)
    # Keep only positive quantities and prices.
    rows_before = len(df)
    valid_values = (
        (df["quantity"] > 0)
        & (df["unit_price"] > 0)
    )
    df = df.loc[valid_values].copy()
    invalid_values_removed = rows_before - len(df)
    # Calculate transaction line value.
    df["line_value"] = (
        df["quantity"] * df["unit_price"]
    )
    # Create useful date features.
    df["invoice_month"] = (
        df["invoice_date"]
        .dt.to_period("M")
        .astype("string")
    )
    df["invoice_year"] = (
        df["invoice_date"].dt.year
    )
    df["invoice_day"] = (
        df["invoice_date"].dt.day
    )
    # Reset row index.
    df = df.reset_index(drop=True)
    metadata = {
        "dataset": "UCI Online Retail II",
        "source_file": RAW_EXCEL_FILE.name,
        "initial_rows": int(initial_rows),
        "reference_rows": int(len(df)),
        "exact_duplicates_removed": int(
            duplicates_removed
        ),
        "cancellations_removed": int(
            cancellations_removed
        ),
        "incomplete_rows_removed": int(
            incomplete_rows_removed
        ),
        "invalid_quantity_or_price_rows_removed": int(
            invalid_values_removed
        ),
        "cleaning_rules": [
            "Remove exact duplicate rows",
            "Exclude cancellation invoices",
            "Remove rows missing essential transaction fields",
            "Keep only positive quantities",
            "Keep only positive unit prices",
            "Calculate line_value as quantity multiplied by unit_price",
            "Derive invoice month, year, and day",
        ],
        "reference_dataset_warning": (
            "The reference dataset is a deterministic cleaned "
            "version of the source data, not guaranteed perfect "
            "ground truth. Cleaning choices may affect analytical "
            "conclusions."
        ),
    }
    return df, metadata
# --------------------------------------------------
# SAVE PROCESSED DATA
# --------------------------------------------------
def save_reference_dataset(
    df: pd.DataFrame,
    metadata: dict,
) -> None:
    """Save the reference dataset, DuckDB table, and metadata."""
    # Save Parquet.
    df.to_parquet(
        REFERENCE_FILE,
        index=False,
    )
    print(f"Reference dataset saved: {REFERENCE_FILE}")
    # Recreate the DuckDB database and reference table.
    if DATABASE_FILE.exists():
        DATABASE_FILE.unlink()
    connection = duckdb.connect(
        str(DATABASE_FILE)
    )
    try:
        connection.execute(
            """
            CREATE TABLE reference_data AS
            SELECT *
            FROM read_parquet(?)
            """,
            [str(REFERENCE_FILE)],
        )
        result = connection.execute(
            "SELECT COUNT(*) FROM reference_data"
        ).fetchone()
        metadata["duckdb_table"] = "reference_data"
        metadata["duckdb_row_count"] = int(result[0])
    finally:
        connection.close()
    print(f"DuckDB database saved: {DATABASE_FILE}")
    # Record the processed dataset hash.
    metadata["reference_file_sha256"] = sha256_file(
        REFERENCE_FILE
    )
    # Write metadata.
    with METADATA_FILE.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Metadata saved: {METADATA_FILE}")
# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("VERITYAI — DATA PREPARATION")
    print("=" * 60)
    raw_df = load_raw_dataset()
    print(f"Raw dataset shape: {raw_df.shape}")
    reference_df, metadata = create_reference_dataset(
        raw_df
    )
    print(
        f"Reference dataset shape: {reference_df.shape}"
    )
    save_reference_dataset(
        reference_df,
        metadata,
    )
    print("\nData preparation completed.")
    print(
        f"Rows before cleaning: "
        f"{metadata['initial_rows']:,}"
    )
    print(
        f"Rows after cleaning: "
        f"{metadata['reference_rows']:,}"
    )
    print(
        f"Exact duplicates removed: "
        f"{metadata['exact_duplicates_removed']:,}"
    )
    print(
        f"Cancellations removed: "
        f"{metadata['cancellations_removed']:,}"
    )
    print(
        f"Incomplete rows removed: "
        f"{metadata['incomplete_rows_removed']:,}"
    )
    print(
        f"Invalid quantity/price rows removed: "
        f"{metadata['invalid_quantity_or_price_rows_removed']:,}"
    )
if __name__ == "__main__":
    main()