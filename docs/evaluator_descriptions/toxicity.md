# ☣️ Toxicity

## Purpose  
The Toxicity Evaluators check whether an LLM-generated response contains harmful, offensive, rude, or toxic language. This is critical when building systems that interact with users in sensitive environments (e.g., support agents, educational tools, or social platforms), where safety and civility are expected.

For example, if your assistant says something sarcastic, aggressive, or demeaning in response to a user's query, this could reflect a failure in prompt handling or model alignment — and should be flagged before deployment.

## How They Work  
The evaluator uses a toxicity classification model (usually based on a transformer trained on toxicity datasets) to assign a **toxicity score** between 0 and 1:

| Toxicity Level      | Score |
|---------------------|-------|
| Non-Toxic           | 0.0   |
| Slightly Toxic      | 0.25  |
| Moderately Toxic    | 0.5   |
| Strongly Toxic      | 0.75  |
| Extremely Toxic     | 1.0   |

The higher the score, the more toxic the response is considered. A score of `0.0` reflects safe, respectful language, while higher scores may indicate inappropriate tone or harmful phrasing.

## Evaluators

### 1. RunToxicityEvaluatorAgainstExpectedScore

This evaluator calculates a toxicity score for a given response and compares it to your expected score and allowed uncertainty. It’s most useful when you want to enforce strict toxicity limits.

**Expected Inputs:**
- `response` - The LLM-generated text to evaluate.
- `expected_score` - The amount of toxicity you are expecting using a score.
- `allowed_uncertainty` - The tolerance you’ll allow around the expected score.

**Results Output:**
- `toxicity` - The calculated toxicity score of the response.
- `response` - The evaluated LLM output.
- `expected_score` - The toxicity score you expected.
- `toxicity_result` - The outcome (`pass` or `fail`) based on comparison.

**When to Use This Evaluator:**

Use this evaluator when:
- You want to strictly limit how toxic an LLM output is allowed to be.
- You're validating outputs in sensitive use cases (e.g., healthcare, children’s apps, moderation tools).
- You’re doing regression testing to ensure new model versions haven’t regressed in safety.

**Example Use Cases:**
- ✅ Ensuring an AI moderator never produces responses with `toxicity > 0.1`.
- ✅ Testing chatbot replies to emotionally charged prompts to confirm they remain respectful.
- ❌ Not ideal if you don’t know what toxicity threshold is acceptable — use golden examples instead.

---

### 2. RunToxicityEvaluatorAgainstReferences

This evaluator compares a response's toxicity score to the average toxicity of a set of golden standard responses. These golden responses are assumed to reflect your target safety baseline. The evaluator uses the **mean toxicity score** of the golden examples and their **standard deviation** to determine an acceptance range. You can control the range using a `scale_uncertainty` factor.

The evaluator will get the mean sentiment score of the golden standards, and use them to calculate their standard deviation, which is used as the uncertainty in the mean score. The standard deviation tells you how spread out the scores are from the mean.

You can scale the uncertainty using any positive float, which adjusts how tightly values must cluster around the mean sentiment score to pass. We recommend scaling between 1 and 3, as this corresponds to standard deviation ranges that cover approximately 68% to 99.7% of values in a normal distribution. Higher values allow for broader acceptance, while lower values enforce stricter confidence around the mean.

**Expected Inputs:**
- `response` - The new response you are evaluating.
- `references` - A list of gold-standard, acceptable responses.
- `scale_uncertainty` - A multiplier to adjust how strict or lenient the tolerance window is.

**Results Output:**
- `toxicity` - The calculated toxicity score of the response.
- `response` - The evaluated output.
- `references` - The gold responses used as references.
- `reference_scores` - Toxicity scores of each golden response.
- `mean_score` - Mean toxicity of the golden responses.
- `calculated_uncertainty` - Standard deviation of the golden scores.
- `toxicity_result` - The outcome (`pass` or `fail`) based on statistical comparison.

**When to Use This Evaluator:**

Use this evaluator when:
- You have examples of "safe" responses but no strict numeric threshold.
- You want to statistically enforce safety alignment to prior curated outputs.
- You’re adjusting or optimizing prompts and want to prevent new toxicity drift.

**Example Use Cases:**
- ✅ Ensuring a newly tuned customer assistant prompt doesn’t introduce sarcasm or passive aggression compared to legacy examples.
- ✅ Verifying automatically rewritten community guidelines responses remain civil.
- ✅ Matching the tone and politeness level of past approved safety-critical responses.
