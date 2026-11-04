from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any
import re
import pandas as pd
@dataclass
class Claim:
    """Structured claim supporting current and legacy interfaces."""
    claim_id: str
    text: str = ""
    claim_type: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    template_id: str = ""
    metric: str = "line_revenue"
    country: str | None = None
    threshold: float = 0.0
    operator: str = ">"
    period_a: str | None = None
    period_b: str | None = None
    top_n: int = 5
    product_code: str | None = None
    tolerance: float = 0.05
    def __post_init__(self):
        if not 1 <= self.top_n <= 20:
            raise ValueError("top_n must be between 1 and 20.")
        if self.operator not in {">", ">=", "<", "<=", "=="}:
            raise ValueError(f"Unsupported operator: {self.operator}")
        if self.template_id and not self.claim_type:
            self.claim_type = self.template_id
        if self.claim_type and not self.template_id:
            self.template_id = self.claim_type
        if self.parameters:
            self.country = self.parameters.get("country", self.country)
            self.threshold = self.parameters.get("threshold", self.threshold)
            self.operator = self.parameters.get("operator", self.operator)
            self.tolerance = self.parameters.get("tolerance", self.tolerance)
            self.period_a = self.parameters.get("period_a", self.period_a)
            self.period_b = self.parameters.get("period_b", self.period_b)
        else:
            self.parameters = {
                "country": self.country,
                "threshold": self.threshold,
                "operator": self.operator,
                "tolerance": self.tolerance,
                "period_a": self.period_a,
                "period_b": self.period_b,
            }
    def model_dump(self) -> dict[str, Any]:
        """Return a deterministic dictionary representation."""
        return asdict(self)
def _revenue_column(df: pd.DataFrame) -> str:
    """Support both historical and current revenue column names."""
    if "line_revenue" in df.columns:
        return "line_revenue"
    if "line_value" in df.columns:
        return "line_value"
    raise ValueError(
        "Dataset requires either 'line_revenue' or 'line_value'."
    )
def make_default_claims(
    df: pd.DataFrame,
    limit: int = 5,
    seed: int = 42,
) -> list[Claim]:
    """Generate a reproducible set of claims."""
    del seed
    revenue_col = _revenue_column(df)
    required = {"country", "invoice_month", revenue_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data = df.copy()
    data[revenue_col] = pd.to_numeric(data[revenue_col], errors="coerce")
    data = data.dropna(subset=["country", "invoice_month", revenue_col])
    claims: list[Claim] = []
    monthly = data.groupby("invoice_month")[revenue_col].sum()
    if not monthly.empty:
        month = str(monthly.idxmax())
        revenue = float(monthly.loc[month])
        claims.append(
            Claim(
                claim_id="monthly_revenue_001",
                text=f"Total revenue in {month} was approximately {revenue:.2f}.",
                claim_type="monthly_revenue",
                metric=revenue_col,
                parameters={
                    "month": month,
                    "threshold": revenue,
                    "tolerance": 0.05,
                },
                tolerance=0.05,
            )
        )
        median_revenue = float(monthly.median())
        claims.append(
            Claim(
                claim_id="monthly_revenue_002",
                text=f"At least one month had revenue greater than {median_revenue:.2f}.",
                claim_type="monthly_revenue_threshold",
                metric=revenue_col,
                parameters={"threshold": median_revenue},
                threshold=median_revenue,
            )
        )
    country_totals = (
        data.groupby("country")[revenue_col]
        .sum()
        .sort_values(ascending=False)
    )
    if not country_totals.empty:
        country = str(country_totals.index[0])
        revenue = float(country_totals.iloc[0])
        claims.append(
            Claim(
                claim_id="country_revenue_001",
                text=f"{country} had the highest recorded revenue among countries.",
                claim_type="top_country_revenue",
                metric=revenue_col,
                country=country,
                parameters={"country": country, "threshold": revenue},
            )
        )
    if "unit_price" in data.columns:
        prices = pd.to_numeric(data["unit_price"], errors="coerce").dropna()
        if not prices.empty:
            average_price = float(prices.mean())
            claims.append(
                Claim(
                    claim_id="average_price_001",
                    text=f"The average unit price was approximately {average_price:.2f}.",
                    claim_type="average_unit_price",
                    metric="unit_price",
                    parameters={
                        "threshold": average_price,
                        "tolerance": 0.05,
                    },
                    tolerance=0.05,
                )
            )
    claims.append(
        Claim(
            claim_id="row_count_001",
            text=f"The dataset contains {len(data)} transaction lines.",
            claim_type="row_count",
            metric=revenue_col,
            parameters={"threshold": len(data)},
            threshold=len(data),
        )
    )
    return claims[:max(0, limit)]
def parse_claim_text(text: str) -> Claim:
    """Parse a supported country-revenue threshold claim."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Claim text must be a non-empty string.")
    pattern = re.compile(
        r"Total revenue for\s+(.+?)\s+was\s+"
        r"(greater than|at least|less than or equal to|less than)\s+"
        r"(-?\d+(?:\.\d+)?)\s*\.?",
        re.IGNORECASE,
    )
    match = pattern.fullmatch(text.strip())
    if not match:
        raise ValueError(
            "Unsupported claim format. Example: "
            "'Total revenue for Germany was greater than 10000.'"
        )
    country, operator_text, threshold_text = match.groups()
    operators = {
        "greater than": ">",
        "at least": ">=",
        "less than": "<",
        "less than or equal to": "<=",
    }
    operator = operators[operator_text.lower()]
    threshold = float(threshold_text)
    return Claim(
        claim_id="parsed_country_revenue",
        text=text.strip(),
        claim_type="country_revenue_threshold",
        metric="line_revenue",
        country=country.strip(),
        threshold=threshold,
        operator=operator,
    )