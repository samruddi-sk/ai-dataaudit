from __future__ import annotations

import json
from datetime import datetime
import os
import re
from functools import lru_cache

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.gemini_extractor import ExtractedClaim, validate_claim


BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "verityai-qwen-lora",
)

SYSTEM_PROMPT = (
    "Extract the structured claim from the question.\n"
    "Return only one valid JSON object. Do not explain your answer.\n"
    "Use these claim types only: country_revenue_threshold, monthly_revenue, "
    "monthly_revenue_threshold, top_country_revenue, average_unit_price, row_count.\n"
    "Use metric revenue, unit_price, or row_count.\n"
    "Preserve comparison operators exactly: >, >=, <, <=, ==.\n"
    "For top_country_revenue, extract top_n; default to 1.\n"
    "Never calculate a result or invent missing information.\n"
)


@lru_cache(maxsize=1)
def load_model():
    if not os.path.isdir(ADAPTER_PATH):
        raise FileNotFoundError(
            f"LoRA adapter not found: {ADAPTER_PATH}"
        )

    print("      Loading fine-tuned Qwen model...")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
        local_files_only=True,
    )

    model.to("cpu")
    model.eval()

    print("      Qwen LoRA model loaded.")
    return tokenizer, model


def extract_json(text: str) -> dict:
    text = text.strip()

    # Remove optional Markdown code fences.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError(
            f"Qwen did not return a valid JSON object: {text[:300]}"
        )

    return json.loads(text[start:end + 1])


def normalize_claim(data: dict, question: str) -> ExtractedClaim:
    aliases = {
        "monthly_sales_revenue_threshold": "monthly_revenue_threshold",
        "monthly_revenue_comparison": "monthly_revenue_threshold",
        "average_item_price": "average_unit_price",
        "top_country": "top_country_revenue",
        "highest_country_revenue": "top_country_revenue",
        "transaction_count": "row_count",
    }

    claim_type = str(data.get("claim_type", "")).strip()
    claim_type = aliases.get(claim_type, claim_type)

    metric = str(data.get("metric", "")).strip().lower()
    metric_aliases = {
        "item_price": "unit_price",
        "average_item_price": "unit_price",
        "sales": "revenue",
        "transactions": "row_count",
        "records": "row_count",
    }
    metric = metric_aliases.get(metric, metric)

    if metric not in {"revenue", "unit_price", "row_count"}:
        metric = {
            "average_unit_price": "unit_price",
            "row_count": "row_count",
        }.get(claim_type, "revenue")

    # Extract the requested number of top countries.
    top_n_match = re.search(
        r"\\btop\\s+(\\d+)\\b",
        question,
        flags=re.IGNORECASE,
    )
    if top_n_match:
        data["top_n"] = int(top_n_match.group(1))

    # Preserve an explicitly written comparison operator.
    operator_match = re.search(
        r"(?<![<>=])(>=|<=|==|>|<)(?![=])",
        question,
    )

    if operator_match:
        data["operator"] = operator_match.group(1)

    # Recover a threshold when the question explicitly contains one.
    number_match = re.search(
        r"(?:>=|<=|==|>|<)\s*([-+]?\d[\d,]*(?:\.\d+)?)",
        question,
    )

    if not number_match:
        number_match = re.search(
            r"\b([-+]?\d[\d,]*(?:\.\d+)?)\s+(?:using|with)\s+(?:>=|<=|==|>|<)",
            question,
            flags=re.IGNORECASE,
        )

    if number_match:
        data["threshold"] = float(
            number_match.group(1).replace(",", "")
        )

    # Normalize natural-language months to the dataset's YYYY-MM format.
    period = data.get("period")
    if period:
        period = str(period).strip()
        for date_format in ("%B %Y", "%b %Y", "%Y-%m"):
            try:
                period = datetime.strptime(
                    period, date_format
                ).strftime("%Y-%m")
                break
            except ValueError:
                continue
        data["period"] = period

    # Use only fields supported by the extraction schema.
    cleaned = {
        "claim_type": claim_type,
        "country": data.get("country"),
        "metric": metric,
        "period": data.get("period"),
        "operator": data.get("operator"),
        "threshold": data.get("threshold"),
        "needs_clarification": bool(
            data.get("needs_clarification", False)
        ),
        "clarification_question": data.get("clarification_question"),
    }

    claim = ExtractedClaim.model_validate(cleaned)
    return validate_claim(claim)


def extract_claim(text: str) -> ExtractedClaim:
    if not text or not text.strip():
        raise ValueError("Claim text cannot be empty.")

    tokenizer, model = load_model()

    prompt = (
        "Extract the structured claim from the question.\n"
        f"Question: {text.strip()}\n"
        "JSON:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=384,
    )

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = output[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    data = extract_json(generated_text)
    return normalize_claim(data, text)
