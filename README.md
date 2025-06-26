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

### 0. Environment Variables

Make sure you have the following environment variables set in your system environment or in a `.env` file located in your project root:

- `AZURE_OPENAI_LLM_MODEL` — The name of the Azure OpenAI language model deployment (e.g., `gpt-35-turbo`).
- `AZURE_OPENAI_LLM_API_KEY` — The API key used to authenticate requests to the language model.
- `AZURE_OPENAI_LLM_ENDPOINT` — The endpoint URL for the Azure OpenAI language model resource.
- `AZURE_OPENAI_LLM_API_VERSION` — The API version to use when calling the language model (e.g., `2024-04-01-preview`).

- `AZURE_OPENAI_EMBEDDING_MODEL` — The name of the Azure OpenAI embedding model deployment.
- `AZURE_OPENAI_EMBEDDING_MODEL_API_KEY` — The API key for authenticating requests to the embedding model.
- `AZURE_OPENAI_EMBEDDING_MODEL_ENDPONT` — The endpoint URL for the embedding model resource.
- `AZURE_OPENAI_EMBEDDING_MODEL_API_VERSION` — The API version used for the embedding model (typically the same as the LLM version).


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
| `similarity`           | `RunSimilarityEvaluator`                       | Embedding-based semantic similarity scoring on a 1–5 scale.                                            | Score between 1 and 5.                                    |
| `similarity`           | `RunSemanticSimilarity`                        | Embedding-based cosine similarity scoring.                                                             | Score between 0.0 and 1.0.                                |
| `similarity`           | `RunMeteorScoreEvaluator`                      | Uses METEOR metric with synonym and stem matching.                                                     | Score between 0.0 and 1.0.                                |
| `similarity`           | `RunBleuScoreEvaluator`                        | BLEU score based on n-gram overlap.                                                                    | Score between 0.0 and 1.0.                                |
| `similarity`           | `RunGleuScoreEvaluator`                        | GLEU score balancing precision and recall.                                                             | Score between 0.0 and 1.0.                                |
| `similarity`           | `RunRougeScoreEvaluator`                       | ROUGE-L F1 score using longest common subsequence.                                                     | Score between 0.0 and 1.0.                                |
| `similarity`           | `RunF1ScoreEvaluator`                          | Token-level precision and recall.                                                                      | Score between 0.0 and 1.0.                                |
| `similarity`           | `RunNonLLMStringSimilarity`                    | Uses string distance metrics like Levenshtein.                                                         | Score between 0.0 and 1.0.                                |
| `similarity`           | `RunStringPresence`                            | Binary evaluator for substring presence.                                                               | 1.0 if found, 0.0 if not.                                 |
| `similarity`           | `RunExactMatch`                                | Binary evaluator for exact match.                                                                      | 1.0 if identical, 0.0 if not.                             |
| `rag`                  | `RunLLMContextPrecisionWithReference`          | Uses LLM to judge how useful retrieved contexts are relative to a reference answer.                    | Score between 0.0 and 1.0.                                |
| `rag`                  | `RunNonLLMContextPrecisionWithReference`       | String-based precision comparing retrieved and reference contexts.                                     | Score between 0.0 and 1.0.                                |
| `rag`                  | `RunLLMContextRecall`                          | LLM-based recall judging how much reference answer is covered by retrieved contexts.                   | Score between 0.0 and 1.0.                                |
| `rag`                  | `RunNonLLMContextRecall`                       | String-similarity based recall comparing reference and retrieved contexts.                             | Score between 0.0 and 1.0.                                |
| `rag`                  | `RunFaithfulness`                              | LLM-based judgment of whether the response is faithful to the retrieved contexts.                      | Score between 0.0 and 1.0.                                |
| `rag`                  | `RunResponseRelevancy`                         | Measures how well the response answers the original query using LLM + embeddings.                      | Score between 0.0 and 1.0.                                |
| `sentiment`            | `RunSentimentEvaluatorAgainstExpectedScore`    | Compares the emotional tone (positive, neutral, negative) of the response against an expected sentiment. | Score between -1 (very negative) and 1 (very positive).   |
| `sentiment`            | `RunSentimentEvaluatorAgainstGoldenStandards`  | Compares the emotional tone of the response against a list of golden standard responses.               | Score between -1 (very negative) and 1 (very positive).   |
| `bias`                 | `RunBiasEvaluatorAgainstExpectedScore`         | Compare the responses potential social, cultural, or political bias against an expected level of bias.  | Score between 0 (neutral) and 1 (biased).                 |
| `bias`                 | `RunBiasEvaluatorAgainstGoldenStandards`       | Compare the responses potential social, cultural, or political bias against golden standard responses.  | Score between 0 (neutral) and 1 (biased).                 |
| `toxicity`             | `RunToxicityEvaluatorAgainstExpectedScore`     | Compare the toxicity (offensive or abusive language) in the response against an expected level of toxicity. | Score between 0 (neutral) and 1 (toxic).                |
| `toxicity`             | `RunToxicityEvaluatorAgainstGoldenStandards`     | Compare the toxicity in the response against a list of golden standards.                             | Score between 0 (neutral) and 1 (toxic).                |
| `format`               | `RunCustomResponseEvaluator`                   | Validates whether the LLM output is in a given format passed to the evaluator.                         | Detected format of the response.                           |
| `format`               | `RunJsonResponseEvaluator`                     | Validates whether the LLM output is in a valid JSON format.                                            | Detected format of the response.                           |


## Which Tool To Use?

## Which Tool To Use?

Here are some flow diagrams that should assist you in picking the right evaluator for the right task.

**I'm looking at the similarity between responses...**

```mermaid
graph TD
  A["Do you want to compare responses for similarity?"] --> B1["Is exact text match required?"]
  B1 --> C1["Yes"] --> D1["Use RunExactMatch"]
  B1 --> C2["No"] --> B2["Is substring presence enough?"]
  B2 --> C3["Yes"] --> D2["Use RunStringPresence"]
  B2 --> C4["No"] --> B3["Do you want semantic similarity using embeddings?"]
  B3 --> C5["Yes"] --> D3["Use RunSimilarityEvaluator or RunSemanticSimilarity"]
  B3 --> C6["No"] --> B4["Do you want BLEU or ROUGE-style string similarity?"]
  B4 --> C7["Yes"] --> D4["Use RunBleuScoreEvaluator, RunRougeScoreEvaluator, or RunF1ScoreEvaluator"]
  B4 --> C8["No"] --> D5["Use RunNonLLMStringSimilarity or RunGleuScoreEvaluator"]
```

**I'm judging the perfomance of my RAG system...**

```mermaid
graph TD
  A["Are you evaluating RAG system behavior?"] --> B1["Are you judging retrieved context quality?"]
  B1 --> C1["Yes"] --> D1["Use RunLLMContextPrecisionWithReference or RunNonLLMContextPrecisionWithReference"]
  B1 --> C2["No"] --> B2["Are you judging whether the response uses the context well?"]
  B2 --> C3["Yes"] --> D2["Use RunFaithfulness or RunLLMContextRecall"]
  B2 --> C4["No"] --> B3["Do you want to check how well the response answers the question?"]
  B3 --> C5["Yes"] --> D3["Use RunResponseRelevancy"]
```

**I'm comparing the sentiment of my responses...**
```mermaid
graph TD
  A["Are you evaluating emotional tone or sentiment?"] --> B1["Do you know the expected sentiment score?"]
  B1 --> C1["Yes"] --> D1["Use RunSentimentEvaluatorAgainstExpectedScore"]
  B1 --> C2["No"] --> B2["Do you have golden responses with the right tone?"]
  B2 --> C3["Yes"] --> D2["Use RunSentimentEvaluatorAgainstGoldenStandards"]
```

**I'm comparing the bias of my responses...**
```mermaid
graph TD
  A["Are you checking for cultural, political, or social bias?"] --> B1["Do you know the maximum acceptable bias score?"]
  B1 --> C1["Yes"] --> D1["Use RunBiasEvaluatorAgainstExpectedScore"]
  B1 --> C2["No"] --> B2["Do you have low-bias golden responses to compare with?"]
  B2 --> C3["Yes"] --> D2["Use RunBiasEvaluatorAgainstGoldenStandards"]
```

**I'm comparing the toxicity of my responses...**
```mermaid
graph TD
  A["Are you checking for toxicity or harmful language?"] --> B1["Do you have a known toxicity threshold?"]
  B1 --> C1["Yes"] --> D1["Use RunToxicityEvaluatorAgainstExpectedScore"]
  B1 --> C2["No"] --> B2["Do you have safe reference responses to compare with?"]
  B2 --> C3["Yes"] --> D2["Use RunToxicityEvaluatorAgainstGoldenStandards"]
```

**I want to check the output format consistency of my responses...**
```mermaid
graph TD
  A["Are you checking response format or structure?"] --> B1["Should the output be valid JSON?"]
  B1 --> C1["Yes"] --> D1["Use RunJsonResponseEvaluator"]
  B1 --> C2["No"] --> B2["Should the output match a specific Python type (e.g., list, dict)?"]
  B2 --> C3["Yes"] --> D2["Use RunCustomResponseEvaluator"]
```

# Things to do...
* Complete extensive user testing
* Test with python version 3.10 and 3.13
* Supress or deal with warnings
* Test with "basics" version of OpenAI API





