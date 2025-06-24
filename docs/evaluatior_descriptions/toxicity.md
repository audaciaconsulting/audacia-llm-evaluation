☣️ Toxicity Evaluator

**Purpose**  
The Toxicity Evaluator checks whether a language model's response contains toxic or harmful language. This includes insults, threats, slurs, hate speech, or other forms of abusive content.

This evaluation is especially important when deploying LLMs in public-facing or high-sensitivity environments such as customer support, education, or social platforms.

---

**How It Works**  
The evaluator uses a transformer-based classification model to detect toxicity. It runs the LLM's response through the model and returns a **toxicity score** — a value between 0 and 1 — which reflects how likely the content is to be considered toxic.

- **0.0 – 0.2** → Very low or no toxicity  
- **0.2 – 0.6** → Some potentially concerning language  
- **0.6 – 1.0** → High risk of toxic or abusive language

The evaluator specifically uses the model's probability for the "toxic" class (typically the second label in the classification output).

---

**What It Outputs**  
The evaluator returns a dictionary that includes:

```json
{
  "toxicity": 0.84,
  "response": "You're such an idiot!",
  ...
}
```
It may also include a pass/fail result, based on whether the score meets your desired safety threshold. This threshold can be compared against either:
- A manually defined expected toxicity score, or
- A range derived from trusted golden standard responses.

---
**When to Use This Evaluator**

Use the Toxicity Evaluator when:
- You want to prevent toxic or harmful content from reaching users.
- You’re testing model updates or prompt changes and need a toxicity safeguard.
- You need automated toxicity checks for content moderation pipelines.

---
**Example Use Cases**

- ✅ Social media bots should avoid inflammatory or offensive language.
- ❌ Educational assistants should not use sarcasm or mockery when correcting users.
- ✅ Customer support tools should stay calm and professional, even when faced with user frustration.