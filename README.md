# Audacia LLM Evaluation Tool
Please note this is a work in progress.

# Notes
* Created initial project structure.
* Use conda and the environment.yml to create the enviornment used in this project, so far only has the python version, update as you add functionality.
* Using project toml to set up package building.

# Things to do...
* Add evaluators to `llm_eval/evaluators.py`
* Add tests to `tests`

## Developement
### Create environment
`pyenv local 3.11.13` 
`uv venv`
`uv sync`

### Update env
`uv add xxx`

### Install package for development
`pip install -e .`

# Evaluators

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







