import logging

from llm_eval.base_evaluators.custom_evaluators import ToxicityEvaluator
from llm_eval.base_evaluators.transformer_evaluators.transformer_base_evaluator import (
    TransformerRunEvaluator,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


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
