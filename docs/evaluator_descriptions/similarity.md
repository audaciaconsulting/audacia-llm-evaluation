# 📏 Similarity Evaluators

## Purpose
Similarity evaluators quantify how closely a model-generated response aligns with a reference output. These tools are essential for evaluating natural language generation tasks like summarization, translation, question answering, and paraphrasing.

They help assess both the lexical overlap and the semantic closeness between outputs, using a mix of token-level, string-level, n-gram, and embedding-based methods.

## How They Work
Each evaluator compares a `response` with a `reference` (or `ground_truth`) using a specific similarity metric. Some rely on classical methods (BLEU, ROUGE), others use embeddings for semantic comparison, and some apply binary or string distance-based techniques.

Results are typically numerical scores with configurable thresholds to decide pass/fail. Some binary methods produce only 0.0/1.0.

## Evaluators

### Summary Table

| Evaluator                     | Method            | Granularity | Measures                                   |
|------------------------------|-------------------|-------------|--------------------------------------------|
| RunSimilarityEvaluator       | Embedding Cosine  | Medium      | Semantic match (0–5 scale)                 |
| RunSemanticSimilarity        | Embedding Cosine  | Medium      | Semantic match (0–1 scale)                 |
| RunMeteorScoreEvaluator      | n-gram + Semantic | Low-Medium  | Lexical and word-level semantic overlap    |
| RunBleuScoreEvaluator        | n-gram            | Low         | Overlap of word sequences                  |
| RunGleuScoreEvaluator        | n-gram            | Low         | Balanced precision/recall overlap          |
| RunRougeScoreEvaluator       | n-gram            | Low         | Summary-level similarity (F1)              |
| RunF1ScoreEvaluator          | Token             | Low         | Precision and recall                       |
| RunNonLLMStringSimilarity    | String Distance   | Low         | String distance metrics (e.g. Levenshtein) |
| RunStringPresence            | String Match      | Low         | Binary presence of reference               |
| RunExactMatch                | String Match      | Low         | Exact match detection                      |

---

### 1. RunSimilarityEvaluator

Uses sentence embeddings to score similarity between the model's output and the reference.

**Expected Inputs:**
- `query` – Context prompt to frame the comparison.
- `response` – Model-generated text.
- `ground_truth` – Reference text to compare against.
- `threshold` – Minimum score (1.0–5.0) to pass.

**Results Output:**
- `similarity` – Score between 1 and 5.
- `similarity_result` – `pass`/`fail`.

**Use When:**
- Semantic alignment matters more than exact wording.
- You want coarse-to-fine-grained judgment across a 5-point scale.

---

### 2. RunSemanticSimilarity

Computes similarity using embeddings and cosine similarity within a [0.0–1.0] scale.

**Expected Inputs:**
- `response` – Model output.
- `reference` – Expected output.
- `threshold` – Minimum cosine similarity (0.0–1.0).

**Results Output:**
- `semantic_similarity` – Score.
- `semantic_similarity_result` – `pass`/`fail`.

**Use When:**
- You want fine-grained semantic comparison with embeddings.
- Compatible with any Langchain-supported embedding model.

---

### 3. RunMeteorScoreEvaluator

Leverages METEOR to account for synonyms, stemming, and order in scoring.

**Expected Inputs:**
- `response` – Generated text.
- `ground_truth` – Reference text.
- `threshold` – METEOR threshold (0.0–1.0).

**Results Output:**
- `meteor` – Score.
- `meteor_result` – `pass`/`fail`.

**Use When:**
- Evaluation requires flexibility in expression (e.g., paraphrasing).

---

### 4. RunBleuScoreEvaluator

Computes BLEU score based on n-gram overlap.

**Expected Inputs:**
- `response` – Generated sentence.
- `ground_truth` – Reference sentence.
- `threshold` – BLEU score threshold (0.0–1.0).

**Results Output:**
- `bleu` – Score.
- `bleu_result` – `pass`/`fail`.

**Use When:**
- Lexical precision is key (e.g., machine translation).

---

### 5. RunGleuScoreEvaluator

GLEU balances precision and recall for n-gram matching.

**Expected Inputs:**
- `response` – Model output.
- `ground_truth` – Gold standard response.
- `threshold` – GLEU threshold (0.0–1.0).

**Results Output:**
- `gleu` – Score.
- `gleu_result` – `pass`/`fail`.

**Use When:**
- Sentence-level evaluation is required with balanced overlap.

---

### 6. RunRougeScoreEvaluator

Uses ROUGE-L (longest common subsequence) to compute F1 scores.

**Expected Inputs:**
- `response` – Generated summary or sentence.
- `ground_truth` – Reference text.
- `threshold` – ROUGE-L F1 threshold (0.0–1.0).

**Results Output:**
- `rouge_f1_score` – Score.
- `rouge_f1_score_result` – `pass`/`fail`.

**Use When:**
- You’re evaluating summarization or gist-level coverage.

---

### 7. RunF1ScoreEvaluator

Token-level comparison using harmonic mean of precision and recall.

**Expected Inputs:**
- `response` – Model output.
- `ground_truth` – Reference string.
- `threshold` – F1 score threshold (0.0–1.0).

**Results Output:**
- `f1` – Score.
- `f1_result` – `pass`/`fail`.

**Use When:**
- You want balanced token overlap accuracy.

---

### 8. RunNonLLMStringSimilarity

Uses string distance metrics (e.g., Levenshtein, Jaro) for similarity.

**Expected Inputs:**
- `response` – Generated response.
- `reference` – Reference string.
- `threshold` – Score threshold (0.0–1.0).

**Results Output:**
- `non_llm_string_similarity` – Score.
- `non_llm_string_similarity_result` – `pass`/`fail`.

**Use When:**
- You prefer character-level distance metrics over semantics.

---

### 9. RunStringPresence

Binary check for whether reference string is present in response.

**Expected Inputs:**
- `response` – Model output.
- `reference` – Substring to match.

**Results Output:**
- `string_presence` – `1.0` if found, `0.0` otherwise.
- `string_presence_result` – `pass`/`fail`.

**Use When:**
- You need guaranteed inclusion of exact wording.

---

### 10. RunExactMatch

Binary evaluator for full-string equality.

**Expected Inputs:**
- `response` – Generated output.
- `reference` – Exact expected output.

**Results Output:**
- `exact_match` – `1.0` or `0.0`.
- `exact_match_result` – `pass`/`fail`.

**Use When:**
- Strict match is required (e.g., classification, ID labels).

---
