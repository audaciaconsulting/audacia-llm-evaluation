# Guidelines for Developing New Evaluators

This document defines the standard structure and design principles for implementing evaluation classes. All evaluators must follow these rules to ensure consistency, maintainability, and interoperability.

---

## ✅ Core Requirements

### 1. Inherit from `Evaluator`

Every evaluator subclasses `Evaluator` and implements one method, `_evaluate()`. The base
provides the public surface — `evaluate()`, `assert_result()`, `__call__()`, and the
`evaluate_async()` / `assert_result_async()` variants — so consumers call every evaluator the
same way.

If the underlying metric is async, subclass `AsyncEvaluator` and implement
`_evaluate_async()` instead. The base runs it to completion, so the public API stays
synchronous: never expose an `async def __call__` or `assert_result`. `RagasBaseEvaluator`
and `BaseScoreEvaluator` already do this.

A synchronous `_evaluate()` may block — a local model, or a network call — without
stalling an async consumer: the base runs it in a thread. Never override
`_evaluate_async` on an `Evaluator` subclass just to make the async path work.

Set `name` (the metric, used in `EvalResult` and logs) and `assertion_fail_message` as class
attributes, or in `__init__` where they vary.

Give the class a docstring covering its purpose, the metric, the granularity level
(Low / Medium / High), and its inputs.

### 2. Return an `EvalResult`

`_evaluate()` returns `EvalResult(name, passed, score, threshold, reason, inputs, raw)`:

- `passed` — whether the criteria were met. This is what `assert_result()` enforces.
- `score` / `threshold` — the number and the bar it had to clear. `threshold` is `None` for
  binary metrics.
- `reason` — why it failed, where the metric explains itself. Appended to the assertion
  message.
- `inputs` — the data that was evaluated.
- `raw` — the flat metric dict, keeping the `{metric}_result` convention:

```json
{
  "response": "This is the output",
  "ground_truth": "This is the reference",
  "bleu_score": 0.76,
  "bleu_result": "pass"
}
```

### 3. Don't reimplement assertion or logging

`assert_result()` raises `AssertionError` with `assertion_fail_message` plus `reason`, and
results are logged for you. Override `_log_payload()` only to shorten a field that is too
large to log.

---

## 🔧 Design Principles

### Prefer Configuration Over Abstraction

To reduce boilerplate and increase readability, child classes should:

- Prefer passing configuration parameters (such as the metric evaluator, result key, or error message) to a reusable base class.
- Implement no more than `_evaluate` (or `_evaluate_async`). A base class may define further
  hooks — `TransformerRunEvaluator._judge`, for instance — but a concrete evaluator should be
  configuration only.

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

## 📦 Dependencies

Some evaluators read the *shape* of what a dependency returns — dict keys from the
azure-ai-evaluation scorers, label/score pairs from the transformers pipeline. The scorer keys
are read with `.get()`, so a rename doesn't raise: the evaluator just reports fail on every
input. That is why `azure-ai-evaluation`, `azure-ai-projects`, `transformers` and `torch` are
capped. Raise a cap deliberately, updating the affected evaluators and tests together.

Select values by name, never by position. The transformers pipeline orders its output by
score, so an index reads whichever label happened to win — a plausible number, silently
wrong. `_extract_score_from_results` looks up `score_label` and raises naming what the model
returned instead.

Run `uv lock --upgrade` regularly — the lock only exercises one point in each declared
range, so check both ends before a release:

```bash
uv sync --resolution lowest-direct && uv run pytest
uv sync --resolution highest && uv run pytest
```



