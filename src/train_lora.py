import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_DIR = Path("data/finetuning")
OUTPUT_DIR = "models/verityai-qwen-lora"

device = "mps" if False else "cpu"
print("Training device:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def load_data(filename):
    records = [
        json.loads(line)
        for line in (DATA_DIR / filename).read_text().splitlines()
        if line.strip()
    ]
    return Dataset.from_list(records)


train_data = load_data("train.jsonl")
test_data = load_data("test.jsonl")


def tokenize(example):
    prompt = (
        "Extract the structured claim from the question.\n"
        f"Question: {example['input']}\n"
        "JSON:"
    )

    answer = example["output"] + tokenizer.eos_token

    prompt_ids = tokenizer(
        prompt, add_special_tokens=False
    )["input_ids"]

    answer_ids = tokenizer(
        answer, add_special_tokens=False
    )["input_ids"]

    max_length = 384
    available_answer_length = max_length - len(prompt_ids)

    answer_ids = answer_ids[:available_answer_length]

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


train_data = train_data.map(
    tokenize, remove_columns=train_data.column_names
)
test_data = test_data.map(
    tokenize, remove_columns=test_data.column_names
)


def collate_features(features):
    max_length = max(len(f["input_ids"]) for f in features)

    input_ids = []
    attention_masks = []
    labels = []

    for feature in features:
        padding = max_length - len(feature["input_ids"])

        input_ids.append(
            feature["input_ids"] + [tokenizer.pad_token_id] * padding
        )
        attention_masks.append(
            feature["attention_mask"] + [0] * padding
        )
        labels.append(
            feature["labels"] + [-100] * padding
        )

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


print("Loading Qwen model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,
)

model.to(device)
model.config.use_cache = False
model.gradient_checkpointing_enable()

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
        use_cpu=True,
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    report_to="none",
    dataloader_pin_memory=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    data_collator=collate_features,
)

print("\nStarting LoRA fine-tuning...")
trainer.train()

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\nFINE-TUNING COMPLETE")
print("Adapter saved to:", OUTPUT_DIR)
