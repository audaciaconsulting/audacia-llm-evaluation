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

Each evaluator returns a dictionary containing a `'result'` field (`'pass'` or `'fail'`), which indicates whether the evaluation meets the expected criteria. Expected criteria can range from user inputted scores, to user inputted golden standard response, or more comparitive elements. All evaluators also include an `assert_result` method for easy unit testing integration. 

Each evaluator may also have additional functionality, for detailed descriptions and configuration options for each evaluator, see the [docs/](docs) directory.


### 1. Importing Evaluators

Each evaluator is accessible via its respective class. For example:

```python
from llm_eval.evaluators.sentiment import RunSentimentEvaluator
```

### 2. Initializing an Evaluator

Instantiate the evaluator with the LLM response you wish to evaluate, plus any other paramters required by the specific evaluator you are using:

```python
response = "I absolutely love this product!"
expected_score = 0.65
allowed_uncertainty = 0.05

evaluator = RunSentimentEvaluator(
    response=response, 
    expected_score=expected_score, 
    allowed_uncertainty=allowed_uncertainty
)
```

### 3. Running the Evaluation

Invoke the evaluator to obtain the evaluation score/result:

```python
result = evaluator()
print(result)
# Output: {'sentiment': 0.62, 'result': 'pass'}
```

### 4. Using the Evaluation Assert

If you're writing a unit test and you want to call the evaluator assert directly, you can use the `assert_result` method built into each evaluator:

```python
def test_sentiment_within_expected_range():
    response = "I absolutely love this product!"
    expected_score = 0.65
    allowed_uncertainty = 0.05
    
    RunSentimentEvaluator(
        response=response, 
        expected_score=expected_score, 
        allowed_uncertainty=allowed_uncertainty
    ).assert_result()
```

# 🧪 Evaluators

The Audacia LLM Evaluation Tool focuses on six key areas of LLM evaluation. In some cases, multiple evaluators are provided for a single area to support varied testing needs and offer greater flexibility and granularity. For full usage documentation, follow the links in the **Description & Documentation** section.

## 📚 Description & Documentation

- [Similarity Scoring](docs/evaluator_descriptions/similarity.md) — Measures how closely an LLM response matches a reference answer.
- [RAG Accuracy](docs/evaluator_descriptions/rag.md) — Evaluates whether the response is factually grounded in retrieved context.
- [Sentiment Scoring](docs/evaluator_descriptions/sentiment.md) — Detects the emotional tone of a response (positive, neutral, negative).
- [Bias Scoring](docs/evaluator_descriptions/bias.md) — Assesses whether a response contains social, cultural, or political bias.
- [Toxicity Scoring](docs/evaluator_descriptions/toxicity.md) — Flags offensive, harmful, or abusive language in the response.
- [Format Consistency](docs/evaluator_descriptions/format.md) — Checks if the response is in the correct structure or JSON format.

## 🔍 Tool Overview

Each `evaluation_tool` belongs to an and `evaluator_area`, and can be accessed via:
```python
from llm_eval.evaluators.evaluator_area import evaluation_tool
```

The table below summarises each evaluator in the Audacia LLM Evaluation Tool, grouped by their target area and purpose:

| Evaluator Area         | Evaluation Tool                                | Description                                                                                            | Basic Output                                              |
|------------------------|------------------------------------------------|--------------------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| `similarity`           |                                                |                                                                                                        |                                                           |
| `rag`                  |                                                |                                                                                                        |                                                           |
| `sentiment`            | `RunSentimentEvaluatorAgainstExpectedScore`    | Compares the emotional tone (positive, neutral, negative) of the response against an expected sentiment. | Score between -1 (very negative) and 1 (very positive).   |
| `sentiment`            | `RunSentimentEvaluatorAgainstGoldenStandards`  | Compares the emotional tone of the response against a list of golden standard responses.               | Score between -1 (very negative) and 1 (very positive).   |
| `bias`                 | `RunBiasEvaluatorAgainstExpectedScore`         | Compare the responses potential social, cultural, or political bias against an expected level of bias.  | Score between 0 (neutral) and 1 (biased).                 |
| `bias`                 | `RunBiasEvaluatorAgainstGoldenStandards`       | Compare the responses potential social, cultural, or political bias against golden standard responses.  | Score between 0 (neutral) and 1 (biased).                 |
| `toxicity`             | `RunToxicityEvaluatorAgainstExpectedScore`     | Compare the toxicity (offensive or abusive language) in the response against an expected level of toxicity. | Score between 0 (neutral) and 1 (toxic).                |
| `toxicity`             | `RunToxicityEvaluatorAgainstGoldenStandards`     | Compare the toxicity in the response against a list of golden standards.                             | Score between 0 (neutral) and 1 (toxic).                |
| `format`               | `RunCustomResponseEvaluator`                   | Validates whether the LLM output is in a given format passed to the evaluator.                         | Detected format of the response.                           |
| `format`               | `RunJsonResponseEvaluator`                     | Validates whether the LLM output is in a valid JSON format.                                            | Detected format of the response.                           |


# Notes
* Created initial project structure.
* Use conda and the environment.yml to create the enviornment used in this project, so far only has the python version, update as you add functionality.
* Using project toml to set up package building.
* All tools now built and functioning.
* Adding documentation.

# Things to do...
* Complete extensive user testing
* Test with python version 3.10 and 3.13
* Supress or deal with warnings





