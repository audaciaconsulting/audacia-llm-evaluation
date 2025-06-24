import logging

from llm_eval.base_evaluators.custom_evaluators import BiasEvaluator
from base_evaluators.transformer_base_evaluator import (
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
