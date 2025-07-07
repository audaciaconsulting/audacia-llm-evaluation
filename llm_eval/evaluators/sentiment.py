import logging

from llm_eval.base_evaluators.custom_evaluators import SentimentEvaluator
from llm_eval.base_evaluators.transformer_base_evaluator import (
    TransformerRunEvaluator,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class RunSentimentEvaluatorAgainstExpectedScore(TransformerRunEvaluator):
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

    def __init__(
        self, response: str, expected_score: float, allowed_uncertainty: float = 0.05
    ):
        super().__init__(
            response=response,
            evaluate_method_args={
                "expected_score": expected_score,
                "allowed_uncertainty": allowed_uncertainty,
            },
            score_key="sentiment",
            evaluator_class=SentimentEvaluator,
            evaluate_method=self.evaluate_against_expected_score,
            assertion_fail_message="Evaluation failed: sentiment of response too different compared to expected score",
        )


class RunSentimentEvaluatorAgainstReferences(TransformerRunEvaluator):
    """
    Sentiment Evaluation Runner Using Golden Standard Comparisons.

    This evaluator applies a transformer-based `SentimentEvaluator` to assess the sentiment of a given
    response and compares the resulting score against a set of golden standard responses.
    It determines pass/fail by checking if the response's sentiment score falls within an acceptable
    range defined by the statistical distribution (mean ± scaled standard deviation) of the
    golden responses.

    For this to work effectively include 10 or more reference responses, 3 is the absolute minimum

    Args:
        response (str): The model-generated response to be evaluated.
        golden_standards (list[str]): A list of reference responses with ideal sentiment.
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
            score_key="sentiment",
            evaluator_class=SentimentEvaluator,
            evaluate_method=self.evaluate_against_responses,
            assertion_fail_message="Evaluation failed: sentiment of response too different compared to golden standard responses",
        )
