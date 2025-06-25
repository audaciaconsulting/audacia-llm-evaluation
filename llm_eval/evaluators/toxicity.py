import logging

from llm_eval.base_evaluators.custom_evaluators import ToxicityEvaluator
from llm_eval.base_evaluators.transformer_base_evaluator import (
    TransformerRunEvaluator,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class RunToxicityEvaluatorAgainstExpectedScore(TransformerRunEvaluator):
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

    def __init__(
        self, response: str, expected_score: float, allowed_uncertainty: float = 0.05
    ):
        evaluate_method_args = {
            "expected_score": expected_score,
            "allowed_uncertainty": allowed_uncertainty,
        }
        super().__init__(response, evaluate_method_args)

    @property
    def evaluator_class(self):
        return ToxicityEvaluator

    @property
    def score_key(self):
        return "toxicity"

    def get_evaluate_method(self) -> type:
        return self.evaluate_against_expected_score


class RunToxicityEvaluatorAgainstGoldenStandards(TransformerRunEvaluator):
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

    def __init__(
        self, response: str, golden_standards: list[str], scale_uncertainty: int = 1
    ):
        super().__init__(
            response=response,
            evaluate_method_args={
                "golden_standards": golden_standards,
                "scale_uncertainty": scale_uncertainty,
            },
        )

    @property
    def evaluator_class(self):
        return ToxicityEvaluator

    @property
    def score_key(self):
        return "toxicity"

    def get_evaluate_method(self) -> type:
        return self.evaluate_against_golden_standards
