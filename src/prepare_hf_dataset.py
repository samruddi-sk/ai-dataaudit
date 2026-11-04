from pathlib import Path
import json
import random

random.seed(42)

out = Path("data/finetuning")
out.mkdir(parents=True, exist_ok=True)

templates = {
    "country_revenue_threshold": [
        "Was revenue in {country} {op} {value}?",
        "Check whether {country} generated {op} {value} in revenue.",
        "Did {country} have revenue {op} {value}?",
        "Is revenue for {country} {op} {value}?",
        "Verify {country} revenue against {value} using {op}.",
    ],
    "monthly_revenue_threshold": [
        "Did any month have revenue {op} {value}?",
        "Was monthly revenue {op} {value}?",
        "Check whether a month's revenue was {op} {value}.",
        "Did monthly sales generate revenue {op} {value}?",
        "Evaluate monthly revenue against {value} using {op}.",
    ],
    "monthly_revenue": [
        "What was the revenue in {month}?",
        "Show total revenue for {month}.",
        "Calculate monthly revenue for {month}.",
        "How much revenue was generated in {month}?",
        "Find sales revenue during {month}.",
    ],
    "top_country_revenue": [
        "Which country generated the highest revenue?",
        "Rank countries by revenue.",
        "Show the top {n} countries by revenue.",
        "Which country had the most sales revenue?",
        "Find the leading country based on revenue.",
    ],
    "average_unit_price": [
        "What was the average unit price?",
        "Calculate the mean selling price per item.",
        "Find the average price of products.",
        "What is the mean unit price across transactions?",
        "Show the average item price.",
    ],
    "row_count": [
        "How many transactions are in the dataset?",
        "Count the total number of records.",
        "What is the total transaction count?",
        "How many rows does the dataset contain?",
        "Give me the number of transactions.",
    ],
}

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


# 1. Country revenue threshold
for i in range(60):
    country = countries[i % len(countries)]
    op = operators[i % len(operators)]
    value = values[i % len(values)]

    add(
        "country_revenue_threshold",
        templates["country_revenue_threshold"][i % 5].format(
            country=country,
            op=op,
            value=value,
        ),
        {
            "country": country,
            "metric": "line_revenue",
            "operator": op,
            "threshold": value,
        },
    )


# 2. Monthly revenue threshold
for i in range(60):
    op = operators[i % len(operators)]
    value = values[(i * 3) % len(values)]

    add(
        "monthly_revenue_threshold",
        templates["monthly_revenue_threshold"][i % 5].format(
            op=op,
            value=value,
        ),
        {
            "metric": "line_revenue",
            "operator": op,
            "threshold": value,
        },
    )


# 3. Monthly revenue
for i in range(60):
    month = months[i % len(months)]

    add(
        "monthly_revenue",
        templates["monthly_revenue"][i % 5].format(month=month),
        {
            "metric": "line_revenue",
            "period": month,
        },
    )


# 4. Top country revenue
for i in range(60):
    template_index = i % 5

    if template_index == 1:
        # Ranking countries: return the top five.
        n = 5
    elif template_index == 2:
        # Explicit top-N question.
        n = (i % 5) + 1
    else:
        # Highest, most, and leading mean one country.
        n = 1

    add(
        "top_country_revenue",
        templates["top_country_revenue"][template_index].format(n=n),
        {
            "metric": "line_revenue",
            "top_n": n,
        },
    )


# 5. Average unit price
for i in range(60):
    add(
        "average_unit_price",
        templates["average_unit_price"][i % 5],
        {
            "metric": "unit_price",
        },
    )


# 6. Row count
for i in range(60):
    add(
        "row_count",
        templates["row_count"][i % 5],
        {
            "metric": "row_count",
        },
    )


random.shuffle(examples)

# Hold out two complete templates per claim type.
train = []
test = []

for kind, phrases in templates.items():
    group = [
        example for example in examples
        if json.loads(example["output"])["claim_type"] == kind
    ]

    held_out = set(phrases[-2:])

    for example in group:
        question = example["input"]

        is_held_out = any(
            question.startswith(phrase.split("{")[0])
            for phrase in held_out
        )

        if is_held_out:
            test.append(example)
        else:
            train.append(example)


random.shuffle(train)
random.shuffle(test)

# Save all datasets.
for filename, rows in [
    ("claim_extraction.jsonl", examples),
    ("train.jsonl", train),
    ("test.jsonl", test),
]:
    with (out / filename).open("w") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


# Validate dataset.
assert len(examples) == 360
assert len(train) == 216
assert len(test) == 144

for example in examples:
    output = json.loads(example["output"])

    if output["claim_type"] == "average_unit_price":
        assert output["metric"] == "unit_price"

    if output["claim_type"] == "top_country_revenue":
        question = example["input"].lower()

        if any(word in question for word in [
            "highest revenue",
            "most sales revenue",
            "leading country",
        ]):
            assert output["top_n"] == 1

    if output["claim_type"] in [
        "country_revenue_threshold",
        "monthly_revenue_threshold",
    ]:
        assert output["operator"] in operators
        assert "threshold" in output
        assert "against" not in output
        assert "alpha" not in output


print("\nDATASET CREATED SUCCESSFULLY")
print("Total:", len(examples))
print("Training:", len(train))
print("Testing:", len(test))

print("\nCLAIM TYPE DISTRIBUTION:")

for kind in templates:
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

print("\nALL DATASET VALIDATION CHECKS PASSED.")
