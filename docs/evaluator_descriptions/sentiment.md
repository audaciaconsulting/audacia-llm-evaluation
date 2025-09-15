# 📊 Sentiment

## Purpose  
The Sentiment Evaluators checks the emotional tone of an LLM-generated response — is it positive, negative, or neutral? This is useful when you're evaluating whether a model is replying in a way that aligns with expectations (e.g., sounding supportive in customer service, or neutral in technical explanations).

For example, if your app expects polite and friendly outputs, a negative sentiment may indicate a problem in the response generation logic.

## How They Work  
The evaluator uses a pre-trained language model (a transformer) to classify the sentiment of the response into one of five categories:

- Very Negative  
- Negative  
- Neutral  
- Positive  
- Very Positive

These categories are then converted into a **numerical sentiment score** using the following weights:

| Sentiment Label  | Score |
|------------------|-------|
| Very Negative    | -1.0  |
| Negative         | -0.5  |
| Neutral          |  0.0  |
| Positive         |  0.5  |
| Very Positive    |  1.0  |

The final score is a single number between -1 and 1 that represents the overall sentiment of the response.

## Evaluators

### 1. RunSentimentEvaluatorAgainstExpectedScore

This evaluator is specifically used for calculating a sentiment score of a response, and then comparing it with your expected sentiment score and allowed uncertainty.

The allowed uncertainty is the range around your expected score that you will allow the response score to fall in and still pass.

For example, if `expected score = 0.5` and `unexpected score = 0.05`, a calculated response score of `sentiment result = 0.52` would pass, whilst `sentiment result = 0.56` would fail.

**Expected Inputs:**
- `response` - This is the LLM response you are evaluating.
- `expected_score` - This is the sentiment score you expect the LLM response to have.
- `allowed_uncertainty` - This is the uncertainty in the expected score you will allow.

**Results Output:**
- `sentiment` - The calculated sentiment of the LLM response.
- `response` - The LLM response passed to the evaluator.
- `expected_score` - The expected score passed to the evaluator.
- `sentiment_result` - The result of the comparison test, either `pass`/`fail`

**When to Use This Evaluator:**

Use this evaluator when:

- You have a clearly defined target sentiment score (e.g., 0.5 for mildly positive) and want to test whether the LLM response falls within an acceptable range of that score.
- You're running regression tests on a single prompt or response to check that tone remains stable over time.
- You want deterministic tests with numerical thresholds for sentiment compliance (e.g., for CI checks or safety gates).

**Example Use Cases:**
- ✅ Ensuring that an AI therapist's replies stay within a friendly and supportive tone (expected_score = 0.5, allowed_uncertainty = 0.1).
- ✅ Checking that a product FAQ generator remains neutral (expected_score = 0.0) across different inputs.
- ❌ Not ideal when you don’t know the precise sentiment score you want — better to use golden responses in that case.

### 2. RunSentimentEvaluatorAgainstReferences

This specific evaluator is used for calculating a sentiment score for an LLM response, and then comparing it with a list of "golden standard" responses. The purpose of this is that you have LLM responses that capture the sentiment you are trying to currently capture, and you want to use them to evaluate your current response without explicitly knowing a score beforehand.

The evaluator will get the mean sentiment score of the golden standards, and use them to calculate their standard deviation, which is used as the uncertainty in the mean score. The standard deviation tells you how spread out the scores are from the mean.

You can scale the uncertainty using any positive float, which adjusts how tightly values must cluster around the mean sentiment score to pass. We recommend scaling between 1 and 3, as this corresponds to standard deviation ranges that cover approximately 68% to 99.7% of values in a normal distribution. Higher values allow for broader acceptance, while lower values enforce stricter confidence around the mean.

**Expected Inputs:**
- `response` - This is the LLM response you are evaluating as a string.
- `references` - This is the list of strings making up your golden standard responses. Ideally 10+ but a minimum of 3 for uncertainty to be calculated.
- `scale_uncertainty` - This is the uncertainty in the expected score you will allow.

**Results Output:**
- `sentiment` - The calculated sentiment of the LLM response.
- `response` - The LLM response passed to the evaluator.
- `references` - The golden standards used to calculated the mean score and the uncertainty.
- `reference_scores` - The individual sentiment scores for each of the golden standards.
- `mean_score` - The calculated mean sentiment score of the golden standards.
- `calculated_uncertainty` - The standard deviation uncertainty of the golden standard scores.
- `sentiment_result` - The result of the comparison test, either `pass`/`fail`

**When to Use This Evaluator:**

Use this evaluator when:
- You don’t have an exact target sentiment score, but you do have example responses that capture the tone you want.
- You’re building or fine-tuning prompts and want to match the tone of previous high-quality outputs.
- You want a flexible and statistical way to enforce sentiment consistency relative to your own best examples.

**Example Use Cases:**
- ✅ Evaluating whether a new assistant prompt maintains the helpful but neutral tone of existing responses.
- ✅ Comparing new model responses to curated golden replies in customer service chats to ensure no tone regression.
- ✅ Validating that generated summaries of user reviews preserve the sentiment distribution of manually-written summaries.

