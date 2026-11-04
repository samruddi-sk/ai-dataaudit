from __future__ import annotations

import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:3b"


def ollama_generate(prompt: str, model: str = DEFAULT_MODEL, timeout: int = 180) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


def paraphrase_claim(claim_text: str, model: str = DEFAULT_MODEL) -> str:
    prompt = (
        "Paraphrase the following data claim without changing its meaning, "
        "numbers, filters, date ranges, or comparison direction. Return only one sentence.\n\n"
        f"Claim: {claim_text}"
    )
    return ollama_generate(prompt, model=model)


def llm_only_judge(claim_text: str, evidence_summary: str, model: str = DEFAULT_MODEL) -> str:
    prompt = (
        "Classify the claim using exactly one label: supported, refuted, "
        "insufficient_evidence, or unreliable_due_to_data_defects. "
        "Do not invent facts. If evidence is insufficient, choose insufficient_evidence.\n\n"
        f"Claim:\n{claim_text}\n\nEvidence summary:\n{evidence_summary}\n\n"
        "Return the label first, then a brief explanation."
    )
    return ollama_generate(prompt, model=model)
