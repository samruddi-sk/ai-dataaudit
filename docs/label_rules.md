# VerityAI Label Rules

## Benchmark item
Each item contains a defect-injected database, a claim, a reference answer from the documented cleaned dataset, and the expected label. The cleaned dataset is an operational reference, not perfect ground truth.

## Label precedence
1. If the claim is undecidable on the reference data (too few rows, undefined calculation, or prespecified confidence interval spanning the threshold), label `insufficient_evidence`.
2. Otherwise, if the injected defect flips the verdict or changes a numerical value beyond the claim's predefined tolerance, label `unreliable_due_to_data_defects`.
3. Otherwise, label `supported` or `refuted` from the reference answer.

A clean reference copy cannot receive `unreliable_due_to_data_defects`.

## Initial claim templates and flip rules
1. **Average transaction value change:** verdict flips when the measured change crosses the claim's threshold. Require at least 30 rows in each comparison period.
2. **Total revenue by country:** verdict flips when revenue crosses the stated monetary threshold.
3. **Top-N products:** verdict flips when the target product enters or leaves the top-N ranking.
4. **Month-over-month growth:** verdict flips when growth crosses the stated threshold or changes direction.

Initial material numerical tolerance is 5% relative change where that makes sense. Store both categorical flips and material numeric shifts separately.

## Repair operators
- `dedupe`: remove exact duplicate rows.
- `drop_cancellations`: remove invoices beginning with C.
- `drop_missing_customer_ids`: exclude records with unknown customer IDs; sensitivity test only, not universally correct.
- `filter_invalid_prices_quantities`: remove non-positive prices or quantities.
- `stratify_by_country`: compare subgroup results to expose aggregation effects.
- `no_repair`: missing transactions are the deliberate no-repair defect. The system must not invent records and should flag uncertainty.

Repairs are plausible scenarios, not guaranteed truth. Document assumptions and retain each variant's metadata.

## Circularity control
Hold out at least one defect type from learned-verifier training. Include a defect with no matching repair operator. Report its performance separately.

## Claim source and parser
Claims should be generated from templates and optionally paraphrased with Ollama `qwen2.5:3b`. Evaluate parser accuracy separately. Never execute unrestricted SQL generated directly by an LLM.
