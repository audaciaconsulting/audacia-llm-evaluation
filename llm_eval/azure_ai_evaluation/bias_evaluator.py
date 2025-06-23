import logging
from statistics import mean, stdev
from typing import List

from llm_eval.custom_evaluator_tools import BiasEvaluator
from llm_eval.utils import format_dict_log

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class RunBiasEvaluator:
    """Evaluates the sentiment of a response and compares it against expectations or standards.

    This class wraps around the `SentimentEvaluator` and provides utility methods to:
    - Get the sentiment score for a given response.
    - Evaluate the score against an expected score with an allowed uncertainty.
    - Evaluate the score against a set of golden standard responses using statistical similarity.

    Attributes:
        response (str): The textual response to be evaluated.

    Methods:
        __call__():
            Returns the sentiment evaluation result for the response.

        evaluate_against_expected_score(expected_score, allowed_uncertainty=0.05):
            Compares the response's sentiment score to an expected score within a given uncertainty range.

        evaluate_against_golden_standards(golden_standards, scale_uncertainty=1):
            Compares the response's sentiment to a set of golden standard responses using statistical measures.
    """

    def __init__(self, response: str):
        """
        Initializes the sentiment evaluation wrapper.

        Args:
            response (str): The textual response to be analyzed.
        """
        self.response = response

    def __call__(self):
        """
        Evaluates the sentiment score of the response.

        Returns:
            dict: A dictionary containing sentiment evaluation results.
        """
        evaluator = BiasEvaluator()
        score = evaluator(
            response=self.response,
        )
        return score

    def evaluate_against_expected_score(
        self, expected_score: float, allowed_uncertainty: float = 0.05
    ):
        """
        Evaluates whether the sentiment score of the response is within an acceptable range of the expected score.

        Args:
            expected_score (float): The target sentiment score.
            allowed_uncertainty (float, optional): The permissible deviation from the expected score.
                Defaults to 0.05.

        Raises:
            AssertionError: If the sentiment score is outside the allowed range.

        Logs:
            Dictionary containing response, expected score, result, and evaluation outcome.
        """
        result = self()
        pass_state = (
            expected_score - allowed_uncertainty
            < result["sentiment"]
            < expected_score + allowed_uncertainty
        )
        result.update(
            {
                "response": self.response,
                "expected_score": expected_score,
                "result": pass_state,
            }
        )
        logger.info(format_dict_log(dictionary=result))
        return result

    def evaluate_against_golden_standards(
        self, golden_standards: List[str], scale_uncertainty: int = 1
    ):
        """
        Evaluates whether the sentiment of the response is within a statistically acceptable range
        compared to a set of golden standard responses.

        Args:
            golden_standards (List[str]): List of exemplar responses with ideal sentiment characteristics.
            scale_uncertainty (int, optional): Multiplier for the standard deviation to define acceptable uncertainty.
                Defaults to 1.

        Raises:
            AssertionError: If the sentiment score is outside the calculated uncertainty range.

        Logs:
            Dictionary containing the response, golden standard scores, mean and uncertainty, and evaluation result.
        """
        current_result = self()
        golden_standard_scores = []
        for response in golden_standards:
            evaluator = BiasEvaluator()
            golden_standard_scores.append(evaluator(response=response)["sentiment"])
        score_mean = mean(golden_standard_scores)
        score_uncertainty = stdev(golden_standard_scores) * scale_uncertainty
        pass_state = (
            score_mean - score_uncertainty
            < current_result["sentiment"]
            < score_mean + score_uncertainty
        )
        current_result.update(
            {
                "response": self.response,
                "golden_standard_example": golden_standards[0],
                "golden_standard_score": score_mean,
                "calculated_uncertainty": score_uncertainty,
                "result": pass_state,
            }
        )
        logger.info(format_dict_log(dictionary=current_result))
        return current_result