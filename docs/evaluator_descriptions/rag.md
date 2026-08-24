# 📊 RAG Evaluators

## Purpose
RAG Evaluators assess the quality of Retrieval-Augmented Generation (RAG) systems across various dimensions. These systems retrieve supporting context before generating an answer, and their success depends on how relevant, accurate, and comprehensive the retrieval and generation are.

These evaluators help you quantify the faithfulness of the response to the retrieved content, measure whether the retrieved contexts were relevant and sufficient, and ensure the generated output remains aligned with the original question.

## How They Work
Each evaluator measures a distinct quality dimension using LLM-based or non-LLM-based metrics. Evaluation is done by passing key inputs — such as the user query, reference answer, generated response, and retrieved contexts — to the corresponding RAG metric implementation.

Each metric outputs a **numerical score between 0 and 1**, with a configurable threshold for determining pass/fail status. The evaluation can use LLMs or simpler string similarity algorithms, depending on the metric type.

## Result

Every evaluator returns an `EvalResult`. Read `passed` for the verdict, `score` and
`threshold` for the number, `reason` where the evaluator explains itself, and `inputs` for
what was evaluated. `raw` carries the flat metric dict, in the `{metric}_result` convention,
for anything the top level does not. Per-evaluator detail below.

## Evaluators

### Summary Table

| Evaluator                                | Retrieval/Generation | Method         | Granularity | Measures                     |
|------------------------------------------|-----------------------|----------------|-------------|------------------------------|
| RunLLMContextPrecisionWithReferenceEvaluator      | Retrieval             | LLM            | High        | Context match to reference   |
| RunNonLLMContextPrecisionWithReferenceEvaluator   | Retrieval             | String Sim.    | Low         | Context text overlap         |
| RunLLMContextRecallEvaluator                      | Retrieval             | LLM            | High        | Recall vs. answer coverage   |
| RunNonLLMContextRecallEvaluator                   | Retrieval             | String Sim.    | High        | Reference context coverage   |
| RunFaithfulnessEvaluator                          | Generation            | LLM            | High        | Truthfulness to context      |
| RunResponseRelevancyEvaluator                     | Generation            | LLM + Embed    | High        | Focus and alignment to query |

All RAG evaluators wrap Ragas metrics, which are async internally, but expose the same synchronous interface as every other evaluator. Use `evaluate_async` / `assert_result_async` if you are already inside an event loop.

### 1. RunLLMContextPrecisionWithReferenceEvaluator

Assesses how well the retrieved contexts align with a reference answer. This precision score uses an LLM to evaluate if top-ranked contexts are actually useful for answering the query.

**Expected Inputs:**
- `user_input` – The original query.
- `reference` – A reference answer that represents the correct response.
- `retrieved_contexts` – A list of retrieved context passages.
- `threshold` – Minimum acceptable precision score (0.0–1.0).

**Result:**
- `passed` – Whether `score` met `threshold`.
- `score` – 0.0–1.0.
- `threshold` – The value you passed.

Also in `raw`: `llm_context_precision_with_reference`,
plus its `_threshold` and `_result` keys, and the inputs.

**Use When:**
- Evaluating retrieval quality relative to known answers.
- Wanting precise LLM judgment of context usefulness.

---

### 2. RunNonLLMContextPrecisionWithReferenceEvaluator

Computes how well the retrieved contexts match a reference set, using string similarity instead of LLMs.

**Expected Inputs:**
- `retrieved_contexts` – Retrieved text passages.
- `reference_contexts` – Known relevant context passages.
- `threshold` – Minimum similarity score (0.0–1.0).

**Result:**
- `passed` – Whether `score` met `threshold`.
- `score` – 0.0–1.0.
- `threshold` – The value you passed.

Also in `raw`: `non_llm_context_precision_with_reference`,
plus its `_threshold` and `_result` keys, and the inputs.

**Use When:**
- You need a lightweight alternative to LLM evaluation.
- Your use case favors exact or near-exact textual matching.

---

### 3. RunLLMContextRecallEvaluator

Evaluates how much of the reference answer is covered by the retrieved contexts using an LLM.

**Expected Inputs:**
- `user_input` – User query.
- `response` – RAG-generated response.
- `reference` – Ground-truth answer.
- `retrieved_contexts` – List of retrieved passages.
- `threshold` – Recall threshold (0.0–1.0).

**Result:**
- `passed` – Whether `score` met `threshold`.
- `score` – 0.0–1.0.
- `threshold` – The value you passed.

Also in `raw`: `llm_context_recall`, plus its `_threshold` and `_result` keys, and the inputs.

**Use When:**
- You want to ensure completeness of retrieval relative to the answer.
- Hallucination prevention is critical.

---

### 4. RunNonLLMContextRecallEvaluator

Measures recall using similarity-based checks between retrieved and reference contexts.

**Expected Inputs:**
- `retrieved_contexts` – List of retrieved contexts.
- `reference_contexts` – List of reference contexts.
- `threshold` – Match threshold (0.0–1.0).

**Result:**
- `passed` – Whether `score` met `threshold`.
- `score` – 0.0–1.0.
- `threshold` – The value you passed.

Also in `raw`: `non_llm_context_recall`, plus its `_threshold` and `_result` keys, and the inputs.

**Use When:**
- Fast, non-LLM recall verification is sufficient.
- Exact coverage comparison is acceptable.

---

### 5. RunFaithfulnessEvaluator

Checks whether the generated response is faithful to the retrieved contexts using LLM entailment reasoning.

**Expected Inputs:**
- `user_input` – The original query.
- `response` – The generated answer.
- `retrieved_contexts` – Supporting documents.
- `threshold` – Minimum faithfulness score (0.0–1.0).

**Result:**
- `passed` – Whether `score` met `threshold`.
- `score` – 0.0–1.0.
- `threshold` – The value you passed.

Also in `raw`: `faithfulness`, plus its `_threshold` and `_result` keys, and the inputs.

**Use When:**
- Hallucination risk needs to be mitigated.
- You want to ensure all response claims are grounded.

---

### 6. RunResponseRelevancyEvaluator

Evaluates how well the generated response addresses the original query. Uses LLM + embeddings to score alignment.

**Expected Inputs:**
- `user_input` – Original prompt or question.
- `response` – Generated response.
- `threshold` – Minimum acceptable relevance score (0.0–1.0).

**Result:**
- `passed` – Whether `score` met `threshold`.
- `score` – 0.0–1.0.
- `threshold` – The value you passed.

Also in `raw`: `response_relevancy`, plus its `_threshold` and `_result` keys, and the inputs.

**Use When:**
- You care about response focus and alignment to query intent.
- Avoiding generic or off-topic answers is important.

---
