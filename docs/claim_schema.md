# Claim Schema

Claims use a Pydantic model with:
- `claim_id`
- `template_id`
- `metric`
- optional `country`
- optional `period_a`, `period_b`
- `threshold`
- `operator`
- `top_n`
- optional `product_code`
- `tolerance`
- `text`
- `source` (`template` or `llm_paraphrased`)

Supported templates:
- `average_value_change`
- `country_revenue_threshold`
- `top_products`
- `month_over_month_growth`

The current text parser intentionally supports only a narrow country-revenue pattern. Expand and test parsing template-by-template. Reject unsupported claims instead of generating unrestricted SQL.
