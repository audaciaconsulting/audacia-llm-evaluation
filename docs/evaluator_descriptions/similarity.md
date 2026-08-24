# 📏 Similarity Evaluators

## Purpose
Similarity evaluators quantify how closely a model-generated response aligns with a reference output. These tools are essential for evaluating natural language generation tasks like summarization, translation, question answering, and paraphrasing.

They help assess both the lexical overlap and the semantic closeness between outputs, using a mix of word-level, string-level, n-gram (sequence of n words), and embedding-based methods.

## How They Work
Each evaluator compares a `response` with a `reference` using a specific similarity metric. Some rely on classical methods (BLEU, ROUGE), others use embeddings for semantic comparison, and some apply binary or string distance-based techniques.

Results are typically numerical scores with configurable thresholds to decide pass/fail. Some binary methods produce only 0.0/1.0.

> **Calling:** every similarity evaluator is synchronous. Some await a scorer internally, but that is not visible from the outside. `evaluate_async` / `assert_result_async` are available for callers already inside an event loop.

## Result

Every evaluator returns an `EvalResult`. Read `passed` for the verdict, `score` and
`threshold` for the number, `reason` where the evaluator explains itself, and `inputs` for
what was evaluated. `raw` carries the flat metric dict, in the `{metric}_result` convention,
for anything the top level does not. Per-evaluator detail below.

## Evaluators

### Summary Table

| Evaluator                     | Method            | Granularity | Measures                                   |
|------------------------------|-------------------|-------------|--------------------------------------------|
| RunSimilarityEvaluator       | LLM Prompt        | High        | Semantic match (1–5 scale)                 |
| RunSemanticSimilarityEvaluator        | Embedding Cosine  | Medium      | Semantic match (0–1 scale)                 |
| RunMeteorScoreEvaluator      | n-gram + Semantic | Low-Medium  | Lexical and word-level semantic overlap    |
| RunBleuScoreEvaluator        | n-gram            | Low         | Overlap of word sequences                  |
| RunGleuScoreEvaluator        | n-gram            | Low         | Balanced precision/recall overlap          |
| RunRougeScoreEvaluator       | n-gram            | Low         | Summary-level similarity (F1)              |
| RunF1ScoreEvaluator          | Word              | Low         | Precision and recall                       |
| RunNonLLMStringSimilarityEvaluator    | String Distance   | Low         | String distance metrics (e.g. Levenshtein) |
| RunStringPresenceEvaluator   | String Match      | Low         | Binary presence of reference               |
| RunExactMatchEvaluator       | String Match      | Low         | Exact match detection                      |

---

### 1. RunSimilarityEvaluator

An LLM scores how semantically aligned the response is with the reference, using the prompt
in `similarity.prompty`. Not embedding cosine similarity — see `RunSemanticSimilarityEvaluator`.

**Expected Inputs:**
- `query` – Context prompt to frame the comparison.
- `response` – Model-generated text.
- `reference` – Reference text to compare against.
- `threshold` – Minimum score to pass. Validated as 0.0–5.0; scores run 1–5.
- `model_config` – Azure OpenAI config for the judge. Defaults to the `LLM_EVAL_*`
  environment variables.

**Result:**
- `passed` – Whether `score` met `threshold`.
- `score` – Between 1 and 5.
- `threshold` – The value you passed.

Also in `raw`: `similarity`, `gpt_similarity` (legacy duplicate), `similarity_threshold`,
`similarity_result`, and the inputs. No `similarity_reason` — the SDK omits it for this metric.

**Use When:**
- Semantic alignment matters more than exact wording.
- You want coarse-to-fine-grained judgment across a 5-point scale.

---

### 2. RunSemanticSimilarityEvaluator

Computes similarity using embeddings and cosine similarity within a [0.0–1.0] scale.

**Expected Inputs:**
- `response` – Model output.
- `reference` – Expected output.
- `threshold` – Minimum cosine similarity (0.0–1.0).

**Result:**
- `passed` – Whether `score` met `threshold`.
- `score` – 0.0–1.0.
- `threshold` – The value you passed.

Also in `raw`: `semantic_similarity`, `semantic_similarity_threshold`,
`semantic_similarity_result`, and the inputs.

**Use When:**
- You want fine-grained semantic comparison with embeddings.
- Compatible with any Langchain-supported embedding model.

---

### 3. RunMeteorScoreEvaluator

Leverages METEOR to account for synonyms, stemming, and order in scoring.

**Expected Inputs:**
- `response` – Generated text.
- `reference` – Reference text.
- `threshold` – METEOR threshold (0.0–1.0).

**Result:**
- `passed` – Whether `score` met `threshold`.
- `score` – 0.0–1.0.
- `threshold` – The value you passed.

Also in `raw`: `meteor_score`, `meteor_threshold`, `meteor_result`, and the inputs.

**Use When:**
- Evaluation requires flexibility in expression (e.g., paraphrasing).

---

### 4. RunBleuScoreEvaluator

Computes BLEU score based on n-gram overlap.

**Expected Inputs:**
- `response` – Generated sentence.
- `reference` – Reference sentence.
- `threshold` – BLEU score threshold (0.0–1.0).

**Result:**
- `passed` – Whether `score` met `threshold`.
- `score` – 0.0–1.0.
- `threshold` – The value you passed.

Also in `raw`: `bleu_score`, `bleu_threshold`, `bleu_result`, and the inputs.

**Use When:**
- Lexical precision is key (e.g., machine translation).

---

### 5. RunGleuScoreEvaluator

GLEU balances precision and recall for n-gram matching.

**Expected Inputs:**
- `response` – Model output.
- `reference` – Gold standard response.
- `threshold` – GLEU threshold (0.0–1.0).

**Result:**
- `passed` – Whether `score` met `threshold`.
- `score` – 0.0–1.0.
- `threshold` – The value you passed.

Also in `raw`: `gleu_score`, `gleu_threshold`, `gleu_result`, and the inputs.

**Use When:**
- Sentence-level evaluation is required with balanced overlap.

---

### 6. RunRougeScoreEvaluator

Uses ROUGE-L (longest common subsequence) to compute F1 scores.

**Expected Inputs:**
- `response` – Generated summary or sentence.
- `reference` – Reference text.
- `threshold` – ROUGE-L F1 threshold (0.0–1.0).

**Result:**
- `passed` – Whether `score` met `threshold`.
- `score` – The ROUGE-L F1 score, 0.0–1.0.
- `threshold` – The value you passed.

Also in `raw`: `rouge_f1_score`, `rouge_precision`, `rouge_recall`, a `_result` and
`_threshold` for each, and the inputs.

**Use When:**
- You’re evaluating summarization or gist-level coverage.

---

### 7. RunF1ScoreEvaluator

Word-level comparison using harmonic mean of precision and recall.

**Expected Inputs:**
- `response` – Model output.
- `reference` – Reference string.
- `threshold` – F1 score threshold (0.0–1.0).

**Result:**
- `passed` – Whether `score` met `threshold`.
- `score` – 0.0–1.0.
- `threshold` – The value you passed.

Also in `raw`: `f1_score`, `f1_threshold`, `f1_result`, and the inputs.

**Use When:**
- You want balanced word overlap accuracy.

---

### 8. RunNonLLMStringSimilarityEvaluator

Uses string distance metrics (e.g., Levenshtein, Jaro) for similarity.

**Expected Inputs:**
- `response` – Generated response.
- `reference` – Reference string.
- `threshold` – Score threshold (0.0–1.0).

**Result:**
- `passed` – Whether `score` met `threshold`.
- `score` – 0.0–1.0.
- `threshold` – The value you passed.

Also in `raw`: `non_llmstring_similarity`, `non_llmstring_similarity_threshold`,
`non_llmstring_similarity_result`, and the inputs.

**Use When:**
- You prefer character-level distance metrics over semantics.

---

### 9. RunStringPresenceEvaluator

Binary check for whether reference string is present in response.

**Expected Inputs:**
- `response` – Model output.
- `reference` – Substring to match.

**Result:**
- `passed` – Whether `score` rounds to 1.
- `score` – `1.0` if the reference is present, `0.0` if not.
- `threshold` – `None`; this evaluator is binary and takes no threshold.

Also in `raw`: `string_presence`, `string_presence_threshold`, `string_presence_result`,
and the inputs.

**Use When:**
- You need guaranteed inclusion of exact wording.

---

### 10. RunExactMatchEvaluator

Binary evaluator for full-string equality.

**Expected Inputs:**
- `response` – Generated output.
- `reference` – Exact expected output.

**Result:**
- `passed` – Whether `score` rounds to 1.
- `score` – `1.0` on an exact match, `0.0` otherwise.
- `threshold` – `None`; this evaluator is binary and takes no threshold.

Also in `raw`: `exact_match`, `exact_match_threshold`, `exact_match_result`, and the inputs.

**Use When:**
- Strict match is required (e.g., classification, ID labels).

---
