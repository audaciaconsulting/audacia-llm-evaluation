import logging

from llm_eval.custom_evaluator_tools import (
    BiasEvaluator,
    SentimentEvaluator,
    ToxicityEvaluator,
)
from llm_eval.transformer_evaluators.transformer_base_evaluator import (
    TransformerRunEvaluator,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class RunBiasEvaluator(TransformerRunEvaluator):
    """
    Evaluation runner for bias detection in LLM responses.

    Uses a transformer-based classifier to assign a bias score to a response, then compares
    it to either an expected score or a set of golden standard responses using uncertainty thresholds.

    Inherits from:
        TransformerRunEvaluator

    Properties:
        evaluator_class: Returns the BiasEvaluator class.
        score_key: Uses "bias" as the key in the result dictionary.
    """
    @property
    def evaluator_class(self):
        return BiasEvaluator

    @property
    def score_key(self):
        return "bias"


class RunSentimentEvaluator(TransformerRunEvaluator):
    """
    Evaluation runner for sentiment analysis in LLM responses.

    Computes an aggregate sentiment score using label weights and validates the result
    against known or gold-standard expectations.

    Inherits from:
        TransformerRunEvaluator

    Properties:
        evaluator_class: Returns the SentimentEvaluator class.
        score_key: Uses "sentiment" as the key in the result dictionary.
    """
    @property
    def evaluator_class(self):
        return SentimentEvaluator

    @property
    def score_key(self):
        return "sentiment"


class RunToxicityEvaluator(TransformerRunEvaluator):
    """
    Evaluation runner for toxicity classification in LLM responses.

    Evaluates whether a given response exhibits toxic content, and validates the
    classification score against a known value or statistical standard.

    Inherits from:
        TransformerRunEvaluator

    Properties:
        evaluator_class: Returns the ToxicityEvaluator class.
        score_key: Uses "toxicity" as the key in the result dictionary.
    """
    @property
    def evaluator_class(self):
        return ToxicityEvaluator

    @property
    def score_key(self):
        return "toxicity"

