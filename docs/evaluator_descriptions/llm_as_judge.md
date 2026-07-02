# ⚖️ LLM-as-Judge Evaluators

## Purpose
LLM-as-Judge evaluators let you define custom evaluation logic in plain language and delegate scoring to a judge model. This is useful when exact-match or standard lexical metrics are too rigid for your use case.

These evaluators are best suited for nuanced checks such as factual correctness, policy compliance, style adherence, and task-specific requirements that are easier to describe than to hard-code.

## How They Work
Use the provided templates in `llm_eval/prompt_templates` for all LLM-as-Judge evaluations.

Each evaluator takes:
- a `prompt` (judge instructions sent as a system message), and
- an `inputs` dictionary (formatted and sent as a human message payload).

The model is invoked with structured output parsing, and the returned payload is converted into a standard result shape.

The prompt **must explicitly instruct** the judge model to return JSON matching the expected schema.

### Template Validation Requirements

The evaluator validates the prompt before running. If you customize a template, you must keep:

- `## Inputs`
- `You will receive:`
- numbered input lines that match `inputs` dict keys exactly and in order (for example `1. query`, `2. response`)
- `## Output Format`
- `Output ONLY valid JSON`
- `llm_as_judge_score`
- `failures_list`

If any required marker is missing, evaluation raises a `ValueError`.

## Evaluators

### Summary Table

| Evaluator                        | Method         | Granularity | Measures                              | Await? |
|----------------------------------|----------------|-------------|---------------------------------------|--------|
| RunLlmAsJudgePassFailEvaluator   | LLM Judge      | High        | Binary pass/fail decision             | No     |
| RunLlmAsJudgeScoreEvaluator      | LLM Judge      | High        | Numeric score with thresholded result | No     |

`Await?` indicates whether `__call__`/`assert_result` return coroutines that must be awaited.

---

### 1. RunLlmAsJudgePassFailEvaluator

Evaluates a prompt and expects the judge to return a binary decision.

**Required Prompt Template:**
- `llm_eval/prompt_templates/llm-as-judge-template.md`

**Expected Inputs:**
- `prompt` - Full evaluation instructions for the judge model.
- `inputs` - Dictionary of fields/values to evaluate (e.g., query, response, reference, context).
- `model` *(optional)* - Custom `AzureChatOpenAI` model; default model is used if omitted.

**Prompt Output Schema (required):**
- `llm_as_judge_score` - `"pass"` or `"fail"`.
- `failures_list` - List of failure reasons.

**Results Output:**
- `llm_as_judge_result` - `pass`/`fail`.
- `failures_list` - Returned failure details.
- `prompt_trunc` - Truncated prompt preview for logging.

**Use When:**
- You want a strict binary verdict.
- The evaluation criteria are custom and prompt-defined.

---

### 2. RunLlmAsJudgeScoreEvaluator

Evaluates a prompt and expects the judge to return a numeric score. The evaluator then converts score to pass/fail using a threshold.

**Required Prompt Template:**
- `llm_eval/prompt_templates/llm-as-judge-score-threshold-template.md`

**Expected Inputs:**
- `prompt` - Full evaluation instructions for the judge model.
- `inputs` - Dictionary of fields/values to evaluate (e.g., query, response, reference, context).
- `threshold` - Minimum score required to pass.
- `model` *(optional)* - Custom `AzureChatOpenAI` model; default model is used if omitted.

**Prompt Output Schema (required):**
- `llm_as_judge_score` - Numeric score.
- `failures_list` - List of failure reasons.

**Results Output:**
- `llm_as_judge_score` - Numeric score returned by judge.
- `llm_as_judge_result` - `pass` when score is `>= threshold`, otherwise `fail`.
- `threshold` - Threshold used for pass/fail conversion.
- `failures_list` - Returned failure details.
- `prompt_trunc` - Truncated prompt preview for logging.

**Use When:**
- You need graded evaluation rather than binary-only output.
- You want configurable strictness using a threshold.

---
