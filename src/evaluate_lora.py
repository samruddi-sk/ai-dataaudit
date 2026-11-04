import json
import torch
import gc
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER = "models/verityai-qwen-lora"
TEST_FILE = "data/finetuning/test.jsonl"
OUTPUT_FILE = "results/lora_evaluation.json"

device = "cpu"


def load_test_data():
    with open(TEST_FILE, "r") as file:
        return [json.loads(line) for line in file if line.strip()]


def extract_json(text):
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        return None
    except Exception:
        return None


def build_prompt(question):
    return (
        "Extract the structured claim from the question.\n"
        f"Question: {question}\n"
        "JSON:"
    )


def evaluate(model, tokenizer, data, label):
    results = []
    model.eval()
    model.to(device)

    for item in tqdm(data, desc=label):
        prompt = build_prompt(item["input"])
        expected = json.loads(item["output"])

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=384
        )

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        generated = output[0][inputs["input_ids"].shape[1]:]
        prediction_text = tokenizer.decode(
            generated,
            skip_special_tokens=True
        )

        prediction = extract_json(prediction_text)

        results.append({
            "input": item["input"],
            "expected": expected,
            "prediction": prediction,
            "raw_output": prediction_text,
            "json_valid": prediction is not None,
            "claim_type_correct": (
                prediction is not None
                and prediction.get("claim_type") == expected.get("claim_type")
            ),
            "fields_correct": prediction == expected
        })

    return results


def metrics(results):
    total = len(results)
    valid = sum(r["json_valid"] for r in results)
    correct_type = sum(r["claim_type_correct"] for r in results)
    exact = sum(r["fields_correct"] for r in results)

    return {
        "total": total,
        "valid_json": valid,
        "json_valid_pct": round(100 * valid / total, 2),
        "claim_type_correct": correct_type,
        "claim_type_accuracy_pct": round(100 * correct_type / total, 2),
        "exact_match": exact,
        "exact_match_pct": round(100 * exact / total, 2)
    }


def main():
    Path("results").mkdir(exist_ok=True)

    data = load_test_data()
    print("Test examples:", len(data))

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print("\nLoading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32
    )

    base_results = evaluate(base, tokenizer, data, "Base model")
    base_metrics = metrics(base_results)

    del base
    gc.collect()

    print("\nLoading fine-tuned model...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32
    )
    tuned = PeftModel.from_pretrained(base, ADAPTER)

    tuned_results = evaluate(tuned, tokenizer, data, "Fine-tuned model")
    tuned_metrics = metrics(tuned_results)

    report = {
        "base_model": base_metrics,
        "fine_tuned_model": tuned_metrics,
        "comparison": {
            "claim_type_accuracy_change_percentage_points": round(
                tuned_metrics["claim_type_accuracy_pct"]
                - base_metrics["claim_type_accuracy_pct"], 2
            ),
            "exact_match_change_percentage_points": round(
                tuned_metrics["exact_match_pct"]
                - base_metrics["exact_match_pct"], 2
            )
        },
        "methodology_note": (
            "Synthetic template-based held-out test set. "
            "Results do not establish real-world generalization."
        ),
        "base_predictions": base_results,
        "fine_tuned_predictions": tuned_results
    }

    with open(OUTPUT_FILE, "w") as file:
        json.dump(report, file, indent=2)

    print("\nBASE MODEL:", json.dumps(base_metrics, indent=2))
    print("\nFINE-TUNED MODEL:", json.dumps(tuned_metrics, indent=2))
    print("\nCOMPARISON:", json.dumps(report["comparison"], indent=2))
    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
