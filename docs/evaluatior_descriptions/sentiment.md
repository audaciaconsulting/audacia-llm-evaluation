# 📊 Sentiment Evaluator

## Purpose  
The Sentiment Evaluator checks the emotional tone of an LLM-generated response — is it positive, negative, or neutral? This is useful when you're evaluating whether a model is replying in a way that aligns with expectations (e.g., sounding supportive in customer service, or neutral in technical explanations).

For example, if your app expects polite and friendly outputs, a negative sentiment may indicate a problem in the response generation logic.

## How It Works  
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

## What It Outputs 
The evaluator returns a dictionary that includes at least the following:

```json
{
  "sentiment": 0.5,
  "response": "Thanks so much, happy to help!"
}
```

Additional fields may include whether the result passes or fails when compared against:
- A manually defined expected sentiment score, or
- A calculated range based on golden standard responses (e.g., trusted or ideal outputs).


## When to Use This Evaluator

Use the Sentiment Evaluator when:
- You want to enforce tone consistency in LLM responses.
- You're testing changes to prompts and want to ensure sentiment hasn’t unintentionally shifted.
- You’re comparing model outputs against gold-standard responses to check tone alignment.


## Example Use Cases
- ✅ Customer support bots should maintain a neutral-to-positive tone.
- ❌ Health advice bots should not sound dismissive or overly casual.
- ✅ Review summarization tools should reflect the sentiment of the original reviews.