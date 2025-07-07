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
    Evaluation runner for sentiment analysis in LLM responses.

    Computes an aggregate sentiment score using label weights and validates the result
    against reference responses.

    Inherits from:
        TransformerRunEvaluator

    Properties:
        evaluator_class: Returns the SentimentEvaluator class.
        score_key: Uses "sentiment" as the key in the result dictionary.
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
