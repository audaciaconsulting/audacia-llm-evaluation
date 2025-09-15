import logging

from llm_eval.base_evaluators.custom_evaluators import BiasEvaluator
from llm_eval.base_evaluators.transformer_base_evaluator import (
    TransformerRunEvaluator,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class RunBiasEvaluatorAgainstExpectedScore(TransformerRunEvaluator):
    """
    Bias Evaluation Runner using Transformer Models.

    This evaluator applies a transformer-based `BiasEvaluator` to assess bias in a given
    response and compares the resulting bias score against an expected value. It is useful
    for validating whether the bias of an LLM-generated response falls within an acceptable
    range of a target score, defined by an uncertainty margin.

    Args:
        response (str): The model-generated response to be evaluated.
        expected_score (float): The expected bias score for comparison.
        allowed_uncertainty (float, optional): Acceptable deviation from the expected score. Defaults to 0.05.
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
            score_key="bias",
            evaluator_class=BiasEvaluator,
            evaluate_method=self.evaluate_against_expected_score,
            assertion_fail_message="Evaluation failed: level of bias in response too different compared to expected score",
        )


class RunBiasEvaluatorAgainstReferences(TransformerRunEvaluator):
    """
    Bias Evaluation Runner Using Golden Standard Comparisons.

    This evaluator applies a transformer-based `BiasEvaluator` to assess the bias of a given
    response and compares the resulting score against a set of golden standard responses.
    It determines pass/fail by checking if the response's bias score falls within an acceptable
    range defined by the statistical distribution (mean ± scaled standard deviation) of the
    golden responses.

    For this to work effectively include 10 or more reference responses, 3 is the absolute minimum

    Args:
        response (str): The model-generated response to be evaluated.
        golden_standards (list[str]): A list of reference responses considered unbiased or ideal.
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
            score_key="bias",
            evaluator_class=BiasEvaluator,
            evaluate_method=self.evaluate_against_responses,
            assertion_fail_message="Evaluation failed: level of bias in response too different compared to reference repsonses",
        )
