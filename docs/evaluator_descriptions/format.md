# 🧾 Format

## Purpose  
The Format Evaluators check whether an LLM-generated response conforms to a required data format — such as a Python type (e.g., `dict`, `list`) or valid JSON. This is essential when your application depends on structured outputs that downstream systems rely on (e.g., calling functions, storing responses, rendering UIs, etc.).

For example, if your LLM is expected to return a dictionary that can be parsed as JSON, any deviation from that format could break your pipeline or API integration.

## How They Work  
Format evaluators determine whether the response returned by the model meets a specific format constraint:

- `RunCustomResponseEvaluator` checks whether the response is an instance of a specified Python type.
- `RunJsonResponseEvaluator` checks whether the response is a valid JSON string that parses into a Python dictionary.

Each evaluator returns a simple pass/fail result along with the original response and the detected format.

## Evaluators

### 1. RunCustomResponseEvaluator

This evaluator verifies whether the response matches a specific Python type (e.g., `dict`, `list`, `str`, etc.). It uses Python's `isinstance()` internally.

**Expected Inputs:**
- `response` - The LLM response to evaluate (any object or string).
- `expected_type` - The Python type that the response is expected to match.
- `assert_result` *(optional)* - If `True`, the evaluator raises an assertion error if the response fails the check.

**Results Output:**
- `response` - The original LLM response.
- `format` - The actual Python type of the response.
- `custom_response_result` - Either `pass` or `fail` based on the type check.

**When to Use This Evaluator:**

Use this evaluator when:
- You're expecting a specific Python type from the model (e.g., a list of items or a dict payload).
- You’re integrating model output into structured systems and want to guard against formatting mismatches.
- You’re debugging or validating typed output from newer function-calling models.

**Example Use Cases:**
- ✅ Checking whether a model-generated function call output is returned as a Python `dict`.
- ✅ Validating that a response representing multiple items is a `list`.
- ✅ Confirming that a model producing markdown content is returning a `str`.

---

### 2. RunJsonResponseEvaluator

This evaluator checks whether a string response is valid JSON and specifically whether it can be parsed into a Python dictionary. It uses `json.loads()` internally and fails if parsing errors occur or the result is not a `dict`.

**Expected Inputs:**
- `response` - The string response to evaluate.
- `assert_result` *(optional)* - If `True`, the evaluator raises an assertion error if parsing fails.

**Results Output:**
- `response` - The original response string.
- `format` - The type after parsing (if successful).
- `json_response_result` - Either `pass` or `fail` depending on the success of parsing and type.

**When to Use This Evaluator:**

Use this evaluator when:
- You expect the LLM output to be a JSON object (`{...}`).
- You want a reliable check that the response can be safely parsed into a `dict`.
- You’re validating model responses for use in APIs, pipelines, or function calls.

**Example Use Cases:**
- ✅ Ensuring a tool-using LLM outputs a valid JSON object for an API call.
- ✅ Validating JSON configuration responses returned by a prompt.
- ✅ Catching format regressions when switching from plain text to structured model outputs.

