# 🧠 Introduction
The **Audacia LLM Evaluation Tool** is a Python package designed to streamline the evaluation of Large Language Model (LLM) outputs. It offers a suite of modular evaluators that assess various aspects of LLM responses, including similarity, retrieval accuracy, sentiment, bias, toxicity, and format consistency.

This tool is ideal for developers, testers, and researchers aiming to:
- **Automate** the evaluation of LLM responses.
- **Benchmark** model outputs against expected results or gold standards.
- **Integrate** evaluation metrics into CI/CD pipelines for continuous monitoring.

Each evaluator operates independently, allowing for flexible integration into diverse workflows. Detailed documentation for each evaluator is available in the docs/ directory.

# 🚀 Getting Started

## 📦 Installation
 
This package currently supports python versions:
- 3.11
- 3.12

To install via cloning the repository:
```bash
# Clone the repository
git clone https://github.com/audaciaconsulting/audacia-llm-evaluation.git
cd audacia-llm-evaluation

# Install the package in editable mode
pip install -e .
``` 

To install directly from github:
```bash
# Install the package from https:
pip install git+https://github.com/audaciaconsulting/audacia-llm-evaluation.git
```

## 🛠️ Usage Guide

Each evaluator returns a dictionary containing the result, which can be used for logging, test assertions, or quality assurance workflows. In most cases, there will also be the option to use the evaluator in `assert mode` allowing it to directly integrate with any unit tests. See the [docs/](docs) folder for in-depth guidance on individual evaluators.

### 1. Importing Evaluators

Each evaluator is accessible via its respective class. For example:

```python
from llm_eval.evaluators.sentiment import RunSentimentEvaluator
```

### 2. Initializing an Evaluator

Instantiate the evaluator with the LLM response you wish to evaluate:

```python
response = "I absolutely love this product!"
evaluator = RunSentimentEvaluator(response=response)
```

### 3. Running the Evaluation

Invoke the evaluator to obtain the evaluation score:

```python
result = evaluator()
print(result)
# Output: {'sentiment': 0.6}
```

### 4. Comparing Against Expected Scores

To validate the evaluation against an expected score:

```python
expected_score = 0.6
evaluation = evaluator.evaluate_against_expected_score(expected_score=expected_score, allowed_uncertainty=0.05)
print(evaluation)
"""
{
    "sentiment": 0.6,
    "response": "I absolutely love this product!",
    "expected_score": 0.6,
    "sentiment_result": "pass"
}
"""
```

### 5. Evaluating Against Golden Standards

To compare the response against a set of gold-standard responses:

```python
golden_responses = [
    "This product is fantastic!",
    "I am very pleased with this item.",
    "Absolutely love it!"
]
evaluation = evaluator.evaluate_against_golden_standards(golden_standards=golden_responses, scale_uncertainty=1)
print(evaluation)
"""
{
    "sentiment": 0.6,
    "response": "I absolutely love this product!",
    "golden_standard_responses: [
        "This product is fantastic!",
        "I am very pleased with this item.",
        "Absolutely love it!"
    ],
    "golden_standard_scores": [0.66, 0.52, 0.62],
    "mean_score": 0.6,
    "calculated_uncertainty": 0.06,
    "sentiment_result": "pass"
}
```

### 6. Handling Format Evaluation

For format-specific evaluations, such as checking if a response is valid JSON:

```python
from llm_eval.evaluators.format import RunFormatEvaluator

response = '{"key": "value"}'
format_evaluator = RunFormatEvaluator(response=response)
format_result = format_evaluator.evaluate_json_response()
print(format_result)
# Output: {'json_response_result': 'pass'}
```

# 🧪 Evaluators

The Audacia LLM Evaluation Tool focuses on six key areas of LLM evaluation. In some cases, multiple evaluators are provided for a single area to support varied testing needs and offer greater flexibility and granularity. For full usage documentation, follow the links in the **Description & Documentation** section.

## 📚 Description & Documentation

- [Similarity Scoring](docs/similarity.md) — Measures how closely an LLM response matches a reference answer.
- [RAG Accuracy](docs/rag.md) — Evaluates whether the response is factually grounded in retrieved context.
- [Sentiment Scoring](docs/sentiment.md) — Detects the emotional tone of a response (positive, neutral, negative).
- [Bias Scoring](docs/bias.md) — Assesses whether a response contains social, cultural, or political bias.
- [Toxicity Scoring](docs/toxicity.md) — Flags offensive, harmful, or abusive language in the response.
- [Format Consistency](docs/format.md) — Checks if the response is in the correct structure or JSON format.

## 🔍 Tool Overview

Each `evaluation_tool` belongs to an and `evaluator_area`, and can be accessed via:
```python
from llm_eval.evaluators.evaluator_area import evaluation_tool
```

The table below summarises each evaluator in the Audacia LLM Evaluation Tool, grouped by their target area and purpose:

| Evaluator Area         | Evaluation Tool          | Description                                                                                   | Basic Output                                              |
|------------------------|--------------------------|-----------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| `similarity`           |                          |                                                                                               |                                                           |
| `rag`                  |                          |                                                                                               |                                                           |
| `sentiment`            | `RunSentimentEvaluator`  | Detects the emotional tone (positive, neutral, negative) of the response.                     | Score between -1 (very negative) and 1 (very positive).   |
| `bias`                 | `RunBiasEvaluator`       | Evaluates the response for potential social, cultural, or political bias.                     | Score between 0 (neutral) and 1 (biased).                 |
| `toxicity`             | `RunToxicityEvaluator`   | Flags toxic, offensive, or abusive language in the response.                                  | Score between 0 (neutral) and 1 (toxic).                  |
| `format`               | `RunFormatEvaluator`     | Validates whether the output is correctly structured, e.g., valid JSON or expected data type. | Format of the response.                                   |


# Notes
* Created initial project structure.
* Use conda and the environment.yml to create the enviornment used in this project, so far only has the python version, update as you add functionality.
* Using project toml to set up package building.
* All tools now built and functioning.
* Adding documentation.

# Things to do...
* Restructure tests
* Complete extensive user testing
* Test with python version 3.10 and 3.13





