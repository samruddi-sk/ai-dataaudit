# AI DataAudit

AI DataAudit is a research prototype for checking whether business claims are supported by structured data and whether results remain stable when data contains common quality defects.

## How it works

1. Load the prepared Online Retail II dataset.
2. Convert a supported business claim into a structured analysis request.
3. Compute results using deterministic Python/SQL logic.
4. Run robustness experiments on datasets with injected defects.
5. Report the evidence and whether the result is stable, sensitive to defects, or insufficient.

Language models may assist with claim understanding or paraphrasing. They do not decide whether a claim is true; verdicts are based on computed evidence.

## Current capabilities

- Supported business metric analysis and country revenue rankings, including top-N rankings.
- Controlled defect experiments, including duplicate rows, missing customer IDs, invalid prices, and missing transaction data.
- Comparison of baseline and defect-affected results.
- Robustness checks for numeric results and top-N ranking changes.
- Streamlit interface for running analyses and reviewing results.
- JSON export of verification results.
- Automated tests with pytest.

This is a constrained research prototype, not a general-purpose text-to-SQL system. Supported claim types and analysis paths are defined by the implementation.

## Technology

- Python, Pandas, and DuckDB
- Streamlit
- Optional local language-model support through Ollama
- Optional Gemini-based claim extraction, when configured
- pytest

## Setup

Python 3.10–3.12 is recommended.

```bash
git clone https://github.com/samruddi-sk/ai-dataaudit.git
cd ai-dataaudit
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Prepare the dataset

Run the repository's data-preparation script:

```bash
python scripts/prepare_data.py
```

The preparation step downloads/processes the UCI Online Retail II data and creates the local analytical database. An internet connection may be required for the initial download.

## Run the application

```bash
streamlit run app.py
```

## Run tests

```bash
python -m pytest -q
```

## Optional language-model setup

Some claim-understanding features can use Gemini API configuration or a local Ollama model. These integrations are optional; evidence calculations and verdict logic are deterministic.

For Ollama, install and start it separately, then pull the model configured by the project before enabling that integration.

## Research limitations

- The system supports a defined set of claim types; it does not understand arbitrary business questions.
- Results depend on the quality and coverage of the source dataset.
- Injected defects are controlled experiments, not a complete model of real-world data corruption.
- Robustness thresholds are experimental settings, not universal statistical guarantees.
- A verification result indicates support within the available data and implemented checks, not proof beyond that evidence.

## Reproducibility

Use the same prepared dataset, project configuration, and defect settings when comparing runs. Record the claim, analysis settings, and exported result so experiments can be repeated.

## License

Add a license before distributing this project for reuse.
