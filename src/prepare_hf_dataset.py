from pathlib import Path
import json
import random
from collections import Counter

random.seed(42)

out = Path("data/finetuning")
out.mkdir(parents=True, exist_ok=True)

countries = [
    "Germany", "France", "United Kingdom",
    "EIRE", "Spain", "Netherlands"
]

operators = [">", ">=", "<", "<="]
values = [1000, 5000, 10000, 25000, 50000, 100000]

months = [
    "December 2010", "January 2011", "February 2011",
    "March 2011", "April 2011", "May 2011",
    "June 2011", "July 2011", "August 2011",
    "September 2011", "October 2011", "November 2011",
]

examples = []


def add(kind, question, fields):
    output = {
        "claim_type": kind,
        **fields,
        "needs_clarification": False,
    }

    examples.append({
        "input": question,
        "output": json.dumps(output),
    })


# 1. Country revenue threshold: 60 unique prompts.
country_templates = [
    "Was revenue in {country} {op} {value}?",
    "Check whether {country} generated {op} {value} in revenue.",
    "Did {country} have revenue {op} {value}?",
    "Is revenue for {country} {op} {value}?",
    "Verify {country} revenue against {value} using {op}.",
]

for i in range(60):
    country = countries[i % len(countries)]
    op = operators[(i // 6) % len(operators)]
    value = values[(i // 4) % len(values)]

    add(
        "country_revenue_threshold",
        country_templates[i % 5].format(
            country=country, op=op, value=value
        ),
        {
            "country": country,
            "metric": "line_revenue",
            "operator": op,
            "threshold": value,
        },
    )


# 2. Monthly revenue threshold: 60 unique prompts.
monthly_threshold_templates = [
    "Did any month have revenue {op} {value}?",
    "Was monthly revenue {op} {value}?",
    "Check whether a month's revenue was {op} {value}.",
    "Did monthly sales generate revenue {op} {value}?",
    "Evaluate monthly revenue against {value} using {op}.",
]

threshold_combinations = [
    (op, value)
    for value in values
    for op in operators
][:12]

for i in range(60):
    op, value = threshold_combinations[i // 5]

    add(
        "monthly_revenue_threshold",
        monthly_threshold_templates[i % 5].format(
            op=op, value=value
        ),
        {
            "metric": "line_revenue",
            "operator": op,
            "threshold": value,
        },
    )


# 3. Monthly revenue: 60 unique prompts.
monthly_templates = [
    "What was the revenue in {month}?",
    "Show total revenue for {month}.",
    "Calculate monthly revenue for {month}.",
    "How much revenue was generated in {month}?",
    "Find sales revenue during {month}.",
]

for i in range(60):
    month = months[i // 5]

    add(
        "monthly_revenue",
        monthly_templates[i % 5].format(month=month),
        {
            "metric": "line_revenue",
            "period": month,
        },
    )


# 4. Top country revenue: 60 unique prompts.
top_templates = [
    "List the top {n} countries by revenue {scope}.",
    "Rank countries by revenue and return the top {n} {scope}.",
    "Show the {n} highest-revenue countries {scope}.",
    "Identify the leading {n} countries by sales revenue {scope}.",
    "Which {n} countries generated the most revenue {scope}?",
    "Display a revenue ranking of the top {n} countries {scope}.",
    "Find the {n} countries with the greatest revenue {scope}.",
    "Return the top {n} countries ranked by total revenue {scope}.",
    "Sort countries by revenue and show {n} results {scope}.",
    "Give me a ranking of the top {n} revenue-generating countries {scope}.",
]

top_scopes = [
    "in the dataset",
    "across all transactions",
    "for the retail data",
    "in the available records",
    "for the complete dataset",
    "based on sales data",
]

for i in range(60):
    n = [1, 3, 5][i % 3]
    template = top_templates[i // 6]
    scope = top_scopes[(i // 3) % 6]

    add(
        "top_country_revenue",
        template.format(n=n, scope=scope),
        {
            "metric": "line_revenue",
            "top_n": n,
        },
    )


# 5. Average unit price: 60 unique prompts.
average_templates = [
    "What was the average unit price {scope}?",
    "Calculate the mean selling price per item {scope}.",
    "Find the average price of products {scope}.",
    "What is the mean unit price across transactions {scope}?",
    "Show the average item price {scope}.",
    "Determine the average unit selling price {scope}.",
    "Compute the mean price per unit {scope}.",
    "What is the typical average selling price per item {scope}?",
    "Calculate the overall average product price {scope}.",
    "Find the mean unit selling price {scope}.",
]

average_scopes = [
    "across all transactions",
    "for the complete dataset",
    "in the retail records",
    "across the full dataset",
    "for all recorded sales",
    "in the available data",
]

for template in average_templates:
    for scope in average_scopes:
        add(
            "average_unit_price",
            template.format(scope=scope),
            {"metric": "unit_price"},
        )


# 6. Row count: 60 unique prompts.
row_templates = [
    "How many transactions are recorded {scope}?",
    "Count the total number of records {scope}.",
    "What is the total transaction count {scope}?",
    "How many rows does the dataset contain {scope}?",
    "Give me the number of transactions {scope}.",
    "Determine the total number of entries {scope}.",
    "Calculate the record count {scope}.",
    "Find how many sales transactions exist {scope}.",
    "What is the total number of data rows {scope}?",
    "Report the number of transactions {scope}.",
]

row_scopes = [
    "in the dataset",
    "across all records",
    "in the retail data",
    "for the complete dataset",
    "in the available data",
    "across the full dataset",
]

for template in row_templates:
    for scope in row_scopes:
        add(
            "row_count",
            template.format(scope=scope),
            {"metric": "row_count"},
        )


# Validate the generated examples before splitting.
assert len(examples) == 360

inputs = [e["input"] for e in examples]
assert len(set(inputs)) == 360, "Duplicate prompts detected!"

# Stratified 60/40 split.
groups = {}

for example in examples:
    output = json.loads(example["output"])
    key = (
        output["claim_type"],
        output.get("top_n"),
    )
    groups.setdefault(key, []).append(example)

train = []
test = []

for group in groups.values():
    random.shuffle(group)
    test_count = round(len(group) * 0.4)

    test.extend(group[:test_count])
    train.extend(group[test_count:])

random.shuffle(train)
random.shuffle(test)


# Final validation.
assert len(train) == 216
assert len(test) == 144

train_inputs = {e["input"] for e in train}
test_inputs = {e["input"] for e in test}

assert not train_inputs.intersection(test_inputs)

for example in examples:
    output = json.loads(example["output"])

    if output["claim_type"] == "average_unit_price":
        assert output["metric"] == "unit_price"

    if output["claim_type"] == "top_country_revenue":
        assert output["top_n"] in {1, 3, 5}

    if output["claim_type"] in {
        "country_revenue_threshold",
        "monthly_revenue_threshold",
    }:
        assert output["operator"] in operators
        assert "threshold" in output


for dataset_name, dataset in [("train", train), ("test", test)]:
    top_n_values = {
        json.loads(e["output"])["top_n"]
        for e in dataset
        if json.loads(e["output"])["claim_type"] == "top_country_revenue"
    }

    assert top_n_values == {1, 3, 5}, (
        f"{dataset_name} is missing a top_n value"
    )


# Save datasets.
for filename, rows in [
    ("claim_extraction.jsonl", examples),
    ("train.jsonl", train),
    ("test.jsonl", test),
]:
    with (out / filename).open("w") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


print("\nDATASET CREATED SUCCESSFULLY")
print("Total:", len(examples))
print("Training:", len(train))
print("Testing:", len(test))
print("Unique prompts:", len(set(inputs)))

print("\nCLAIM TYPE DISTRIBUTION:")

for kind in [
    "country_revenue_threshold",
    "monthly_revenue_threshold",
    "monthly_revenue",
    "top_country_revenue",
    "average_unit_price",
    "row_count",
]:
    total = sum(
        json.loads(e["output"])["claim_type"] == kind
        for e in examples
    )
    tr = sum(
        json.loads(e["output"])["claim_type"] == kind
        for e in train
    )
    te = sum(
        json.loads(e["output"])["claim_type"] == kind
        for e in test
    )

    print(f"{kind}: total={total}, train={tr}, test={te}")

print("\nTOP_N DISTRIBUTION:")

for name, dataset in [("train", train), ("test", test)]:
    counts = Counter(
        json.loads(e["output"])["top_n"]
        for e in dataset
        if json.loads(e["output"])["claim_type"] == "top_country_revenue"
    )
    print(f"{name}: {dict(sorted(counts.items()))}")

print("\nALL VALIDATION CHECKS PASSED.")
