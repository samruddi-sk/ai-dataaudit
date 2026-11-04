from pathlib import Path
import json
import random
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/finetuning/claim_extraction.jsonl"
OUTPUT = ROOT / "data/finetuning"

random.seed(42)


def main():
    groups = defaultdict(list)

    with INPUT.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            item = json.loads(line)

            if "input" not in item or "output" not in item:
                raise ValueError(f"Invalid example at line {line_number}")

            output = json.loads(item["output"])
            groups[output["claim_type"]].append(item)

    train_data = []
    test_data = []

    for claim_type, examples in groups.items():
        random.shuffle(examples)

        if len(examples) < 2:
            raise ValueError(
                f"Not enough examples for {claim_type}"
            )

        test_count = max(1, round(len(examples) * 0.2))

        test_data.extend(examples[:test_count])
        train_data.extend(examples[test_count:])

    random.shuffle(train_data)
    random.shuffle(test_data)

    def save_jsonl(filename, data):
        with (OUTPUT / filename).open("w", encoding="utf-8") as file:
            for item in data:
                file.write(json.dumps(item, ensure_ascii=False) + "\n")

    save_jsonl("train.jsonl", train_data)
    save_jsonl("test.jsonl", test_data)

    print("\nDATASET SPLIT COMPLETE")
    print("=" * 45)
    print(f"Training examples: {len(train_data)}")
    print(f"Testing examples:  {len(test_data)}")
    print(f"Total examples:    {len(train_data) + len(test_data)}")

    print("\nTraining distribution:")
    for claim_type in sorted(groups):
        print(
            f"{claim_type}: "
            f"{sum(json.loads(x['output'])['claim_type'] == claim_type for x in train_data)}"
        )

    print("\nFiles created:")
    print(OUTPUT / "train.jsonl")
    print(OUTPUT / "test.jsonl")


if __name__ == "__main__":
    main()
