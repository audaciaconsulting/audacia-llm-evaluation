# ⚖️ Bias

## Purpose  
The Bias Evaluators check whether an LLM-generated response exhibits biased, prejudiced, or unbalanced language. This is useful when you're assessing whether a model's outputs are fair, especially in sensitive applications (e.g., hiring tools, education platforms, or healthcare assistants).

For example, if your app is expected to provide unbiased descriptions of people or scenarios, an answer that unfairly favors one group or viewpoint may indicate an issue with prompt design, training data, or model selection.

## How They Work  
The evaluator uses a transformer-based bias classifier to detect how biased a given response is. It produces a **numerical bias score** between 0 and 1:

| Bias Level         | Score |
|--------------------|-------|
| No Bias            | 0.0   |
| Slight Bias        | 0.25  |
| Moderate Bias      | 0.5   |
| Strong Bias        | 0.75  |
| Very Strong Bias   | 1.0   |

The final score reflects the severity of bias present in the LLM output. Lower scores indicate more neutral responses, while higher scores suggest more problematic or skewed language.

## Evaluators

### 1. RunBiasEvaluatorAgainstExpectedScore

This evaluator calculates a bias score for a given response and compares it to your expected score and a user-defined uncertainty range. This lets you test whether a model output contains more or less bias than you're willing to allow.

**Expected Inputs:**
- `response` - The LLM-generated output you're evaluating.
- `expected_score` - The expected bias for this output as a numerical score.
- `allowed_uncertainty` - The amount of deviation you’ll tolerate from the expected score.

**Results Output:**
- `bias` - The calculated bias score of the LLM response.
- `response` - The LLM response evaluated.
- `expected_score` - The target score you were testing against.
- `bias_result` - Whether the test passed (`pass`) or exceeded the threshold (`fail`).

**When to Use This Evaluator:**

Use this evaluator when:
- You have a clear tolerance level for bias and want to automatically reject outputs that exceed it.
- You’re running regression tests to ensure changes don’t introduce additional bias.
- You're monitoring fairness in content generation pipelines (e.g., summarizers, classifiers).

**Example Use Cases:**
- ✅ Enforcing a strict bias threshold on AI-generated student feedback to ensure fairness.
- ✅ Testing new prompts for a hiring assistant to confirm no output scores above `expected_score = 0.25`.
- ❌ Not ideal if you lack a clear numeric bias target — use golden examples instead.

---

### 2. RunBiasEvaluatorAgainstReferences

This evaluator compares the bias score of a generated response against a set of golden standard responses that reflect your preferred (low-bias) tone. It computes the **mean bias score** from your golden samples, uses the **standard deviation** as uncertainty, and checks whether the new response falls within this statistical range.

The evaluator will get the mean sentiment score of the golden standards, and use them to calculate their standard deviation, which is used as the uncertainty in the mean score. The standard deviation tells you how spread out the scores are from the mean.

You can scale the uncertainty using any positive float, which adjusts how tightly values must cluster around the mean sentiment score to pass. We recommend scaling between 1 and 3, as this corresponds to standard deviation ranges that cover approximately 68% to 99.7% of values in a normal distribution. Higher values allow for broader acceptance, while lower values enforce stricter confidence around the mean.

**Expected Inputs:**
- `response` - The new LLM output to evaluate.
- `golden_standards` - A list of ideal, low-bias outputs used as the benchmark.
- `scale_uncertainty` - A multiplier to control strictness around the golden mean score.

**Results Output:**
- `bias` - The calculated bias score of the new response.
- `response` - The evaluated LLM response.
- `golden_standard_responses` - The golden examples used for comparison.
- `golden_standard_scores` - Individual bias scores for each golden response.
- `mean_score` - Average bias score of the golden standards.
- `calculated_uncertainty` - The standard deviation across golden scores.
- `bias_result` - Whether the response passed (`pass`) or exceeded the acceptable range (`fail`).

**When to Use This Evaluator:**

Use this evaluator when:
- You want to compare current responses to high-quality, low-bias golden outputs.
- You don't know the "right" score ahead of time but do have reference examples.
- You’re iterating on prompt design and need quick feedback on fairness drift.

**Example Use Cases:**
- ✅ Ensuring a model-generated biography doesn’t deviate in tone from known unbiased samples.
- ✅ Comparing political commentary responses against a curated set of neutral golden standards.
- ✅ Validating AI tutors’ answers remain as balanced as historical teaching samples.
