# AI DataAudit
**Repair-Robust Verification of AI-Generated Claims Under Data Quality Defects**

AI DataAudit is a research prototype that checks structured claims against transaction data, tests sensitivity to plausible data repairs, and abstains when evidence is insufficient or defect-sensitive.

## What is included
- Reproducible data preparation for UCI Online Retail II
- DuckDB reference database
- Four predefined claim templates
- Controlled defect injection
- Rule-based repair-robustness verification
- Optional Ollama `qwen2.5:3b` claim paraphrasing and LLM-only baseline
- CSV-based benchmark results and metrics
- Thin Streamlit interface
- Pytest tests and research design documentation

## Important research caveat
This is a **working research prototype**, not evidence that the method is novel or publishable. The cleaned reference data is an operational reference, not perfect ground truth. Labels and repairs are based on documented assumptions. Validate the method against related literature before claiming novelty.

## 1. Requirements
- Python 3.10–3.12 recommended
- Internet access for initial dataset download
- Optional: Ollama installed locally for LLM features

## 2. Setup on macOS / VS Code

```bash
cd ai-dataaudit
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Prepare the dataset

```bash
python -m src.prepare_data
```

This downloads UCI Online Retail II through `ucimlrepo`, saves a raw CSV, creates a documented reference Parquet file and DuckDB database, and records SHA-256 hashes in `data/processed/metadata.json`.

The reference cleaning rules:
- remove exact duplicate rows;
- exclude cancellation invoices (invoice numbers beginning with `C`);
- parse dates and numeric columns;
- remove rows missing core transaction fields;
- exclude non-positive quantities or unit prices;
- retain missing customer IDs as unknown attributes.

This is a benchmark baseline, not a claim that the underlying dataset is perfectly clean. Inspect `metadata.json` after running.

## 4. Run a small benchmark

```bash
python -m src.run_benchmark --claims 120 --seed 42
```

Results are written to `results/benchmark_results.csv` and a summary to `results/benchmark_summary.json`. The initial benchmark is synthetic and intended for debugging and preliminary analysis.

## 5. Run tests

```bash
pytest -q
```

## 6. Launch the demo

```bash
streamlit run app.py
```

The app loads the local reference database. It supports a small set of constrained claim templates; it is not a general-purpose text-to-SQL system.

## 7. Optional Ollama integration

Install Ollama, then:

```bash
ollama pull qwen2.5:3b
```

Use `--llm-sample 30` for a small local LLM baseline. Avoid running thousands of prompts on an 8 GB machine.

```bash
python -m src.run_benchmark --claims 120 --llm-sample 30 --seed 42
```

Ollama integration is optional. The deterministic baseline and tests work without it.

## 8. Reproducibility
- Use a fixed seed with `--seed`.
- Raw and reference data hashes are recorded in metadata.
- Benchmark results include the seed and benchmark settings.
- Generated data and results are excluded from Git by default; keep them locally for experiments and selectively publish small result files if appropriate.

## Project structure

```text
ai-dataaudit/
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
├── src/
│   ├── __init__.py
│   ├── prepare_data.py
│   ├── claims.py
│   ├── defects.py
│   ├── verifier.py
│   ├── llm.py
│   └── run_benchmark.py
├── tests/
│   ├── test_claims.py
│   ├── test_defects.py
│   └── test_verifier.py
├── docs/
│   ├── label_rules.md
│   ├── claim_schema.md
│   ├── related_work.md
│   └── dataset_card.md
├── data/
│   ├── raw/
│   └── processed/
└── results/
```
