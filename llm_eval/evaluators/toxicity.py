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
        super().__init__(
            response=response,
            evaluate_method_args={
                "expected_score": expected_score,
                "allowed_uncertainty": allowed_uncertainty,
            },
            score_key="toxicity",
            evaluator_class=ToxicityEvaluator,
            evaluate_method=self.evaluate_against_expected_score,
            assertion_fail_message="Evaluation failed: toxicity of response too different compared to expected score",
        )


class RunToxicityEvaluatorAgainstReferences(TransformerRunEvaluator):
    """
    Toxicity Evaluation Runner Using Golden Standard Comparisons.

    This evaluator applies a transformer-based `ToxicityEvaluator` to assess the toxicity of a given
    response and compares the resulting score against a set of golden standard responses.
    It determines pass/fail by checking if the response's toxicity score falls within an acceptable
    range defined by the statistical distribution (mean ± scaled standard deviation) of the
    golden responses.

    For this to work effectively include 10 or more reference responses, 3 is the absolute minimum

    Args:
        response (str): The model-generated response to be evaluated.
        golden_standards (list[str]): A list of reference responses with no or ideal toxicity.
        scale_uncertainty (int, optional): Scaling factor for standard deviation used to calculate
            the tolerance range. Defaults to 1.
    """

    def __init__(
        self, response: str, references: list[str], scale_uncertainty: int = 1
    ):
        super().__init__(
            response=response,
            evaluate_method_args={
                "references": references,
                "scale_uncertainty": scale_uncertainty,
            },
            score_key="toxicity",
            evaluator_class=ToxicityEvaluator,
            evaluate_method=self.evaluate_against_responses,
            assertion_fail_message="Evaluation failed: sentiment of response too different compared to golden standard responses",
        )
