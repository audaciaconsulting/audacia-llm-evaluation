from typing import Any
from transformers import pipeline

from llm_eval.model_tools import REQUIRED_MODELS


class SentimentEvaluator:
    def __init__(self):
        pass

    def __call__(self, *, response: str, **kwargs):
        WEIGHTS = {
            "Very Negative": -1.0,
            "Negative": -0.5,
            "Neutral": 0.0,
            "Positive": 0.5,
            "Very Positive": 1.0,
        }
        # set up the scoring pipeline
        sentiment_pipe = pipeline(
            "text-classification",
            model=REQUIRED_MODELS["sentiment"]['name'],
            return_all_scores=True,
            device="cpu",
        )
        # score the AI system response
        full_sentiment = sentiment_pipe(response)[0]
        agg_sentiment_score = sum(
            WEIGHTS[x["label"]] * x["score"] for x in full_sentiment
        )
        return {"sentiment": agg_sentiment_score}


class FormatEvaluator:
    def __init__(self):
        pass

    def __call__(self, *, response: Any, **kwargs):
        return {"format": type(response)}