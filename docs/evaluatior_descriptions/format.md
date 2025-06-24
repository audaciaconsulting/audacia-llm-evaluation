📐 Format Evaluator

**Purpose**  
The Format Evaluator checks whether a language model's response is returned in the correct **data format or structure** — for example, whether it’s a string, dictionary, list, or well-formed JSON.

This is especially useful in automated systems or API integrations, where responses must conform to a specific schema or structure (e.g., a JSON object) in order to be parsed or used programmatically.

---

**How It Works**  
The Format Evaluator offers three main capabilities:

1. **Detect the response type**  
   Automatically identifies and logs the type of the response (e.g., `str`, `dict`, `list`).

2. **Check against an expected Python type**  
   Validates whether the response is of a specific type (e.g., `dict` or `str`), and returns a **pass/fail** result.

3. **Validate JSON format**  
   Tries to parse the response as JSON and checks that it results in a dictionary (a common format for structured data). Useful for verifying that the model returned valid JSON output.

---

**What It Outputs**  
When you run the evaluator, you’ll get results like:

```json
{
  "format": "<class 'str'>"
}
```
For specific validation checks, such as custom type or JSON structure, you’ll receive:
```json
{
  "custom_response_result": "pass"
}
```
or:
```json
{
  "json_response_result": "fail"
}
```
---
**When to Use This Evaluator**

Use the Format Evaluator when:
- You need to ensure LLM responses match a required structure (e.g., valid JSON objects).
- You’re building integrations where incorrect formats could break downstream logic.
- You want to enforce consistent response types during testing or fine-tuning.
---
**Example Use Cases**

- ✅ API assistants should return a valid JSON object like {"status": "ok", "data": {...}}.
- ❌ A response expected to be a list of results should not return a plain string.
- ✅ Form generators should always output dictionaries representing field names and values.