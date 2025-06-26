# Guidelines for Developing New Evaluators

This document defines the standard structure and design principles for implementing evaluation classes. All evaluators must follow these rules to ensure consistency, maintainability, and interoperability.

---

## ✅ Core Requirements

### 1. Class-Based Implementation

Each evaluator **must** be implemented as a Python class and provide:

- A `__call__()` method that performs the evaluation.
- An `assert_result()` method that raises an `AssertionError` on failure.
- A clear class-level docstring describing:
  - The purpose of the evaluator.
  - The evaluation method or metric.
  - The granularity level (Low / Medium / High).
  - The expected inputs and structure of outputs.

---

### 2. `__call__()` Method

This method is responsible for performing the evaluation and returning a dictionary result. It **must**:

- Include all relevant inputs (e.g., `response`, `ground_truth`) in the returned dictionary.
- Contain metric values used in evaluation.
- Include a `{metric}_result` field set to `"pass"` or `"fail"`.

#### Example Return Format

```json
{
  "response": "This is the output",
  "ground_truth": "This is the reference",
  "bleu_score": 0.76,
  "bleu_result": "pass"
}
```
### 3. `assert_result()` Method

This method is responsible for enforcing evaluation correctness. It must:

- Invoke the `__call__()` method.
- Check the `{metric}_result` field in the result dictionary.
- Raise an `AssertionError` with a clear and informative message if the result is `"fail"`.

---

## 🔧 Design Principles

### Prefer Configuration Over Abstraction

To reduce boilerplate and increase readability, child classes should:

- Prefer passing configuration parameters (such as the metric evaluator, result key, or error message) to a reusable base class.
- Avoid implementing abstract methods unless necessary.

#### ✅ Example

```python
class RunF1ScoreEvaluator(BaseScoreEvaluator):
    def __init__(self, response: str, ground_truth: str, threshold: float):
        evaluator = F1ScoreEvaluator(threshold=threshold)
        super().__init__(
            response=response,
            ground_truth=ground_truth,
            threshold=threshold,
            result_key="f1_result",
            evaluator=evaluator,
            assertion_fail_message="Evaluation failed: F1 score below threshold"
        )
```
## 📌 Naming Convention

Evaluator classes should follow this naming pattern:  
`Run<MetricName>Evaluator`

### Examples

- `RunBleuScoreEvaluator`
- `RunExactMatchEvaluator`
- `RunBiasEvaluatorAgainstExpectedScore`



