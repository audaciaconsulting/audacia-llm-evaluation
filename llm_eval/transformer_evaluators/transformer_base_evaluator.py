import logging
from abc import ABC, abstractmethod
from statistics import mean, stdev
from typing import List

from llm_eval.utils import format_dict_log

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TransformerRunEvaluator(ABC):
    """
    Abstract base class for running transformer-based evaluation on a response and validating
    it against expectations or golden standards.

    This class defines a reusable evaluation interface for subclasses like sentiment, bias,
    or toxicity evaluators that wrap specific Transformer models and scoring logic.

    Attributes:
        response (str): The textual response to be evaluated.

    Subclasses must define:
        - evaluator_class (a property returning the appropriate evaluator class)
        - score_key (a property indicating which key in the result contains the score)

    Example:
        class RunSentimentEvaluator(TransformerRunEvaluator):
            @property
            def evaluator_class(self):
                return SentimentEvaluator

            @property
            def score_key(self):
                return "sentiment"
    """

    def __init__(self, response: str, evaluate_method: type, evaluate_method_args: dict):
        """
        Initializes the evaluator with a response.

        Args:
            response (str): The textual response to evaluate.
        """
        self.response = response
        self.evaluate_method = evaluate_method
        self.evaluate_method_args = evaluate_method_args
        self.result = None

    @property
    @abstractmethod
    def evaluator_class(self):
        """
        Returns the evaluator class used for scoring.

        Must be implemented by subclasses.

        Returns:
            Callable: A class that takes a response and returns a score dictionary.
        """
        pass

    @property
    @abstractmethod
    def score_key(self):
        """
        Returns the key used to extract the score from the result dictionary.

        Must be implemented by subclasses.

        Returns:
            str: The name of the score field (e.g., "sentiment", "bias").
        """
        pass

    def __call__(self):
        """
        Runs the evaluation on the response using the configured evaluator class.

        Returns:
            dict: A dictionary containing the evaluation score.
        """
        evaluator = self.evaluator_class()
        self.result = evaluator(response=self.response)
        return self.evaluate_method(**self.evaluate_method_args)

    def _assert_result(self, result: dict, message: str):
        """
        Raises an AssertionError if the result is not a pass.

        Args:
            result (dict): The result dictionary with a boolean `result` key.
            message (str): The error message to include if assertion fails.

        Raises:
            AssertionError: If result['result'] is False or missing.
        """
        if result.get("result") == 'fail':
            raise AssertionError(message)

    def evaluate(self, assert_result=False):
        self.result = self()
        logger.info(format_dict_log(dictionary=self.result))

        if assert_result:
            self._assert_result(
                self.result, f"Evaluation failed for {self.score_key} against known score"
            )

        return self.result

    def evaluate_against_expected_score(
            self,
            expected_score: float,
            allowed_uncertainty: float = 0.05
    ):
        """
        Compares the evaluated score to an expected score within a given uncertainty margin.

        Args:
            expected_score (float): The target score the response should be close to.
            allowed_uncertainty (float, optional): Acceptable deviation from the expected score. Defaults to 0.05.
            assert_result (bool, optional): Whether to raise an error on failure. Defaults to False.

        Returns:
            dict: A dictionary including evaluation result, expected score, and pass/fail flag.

        Raises:
            AssertionError: If the result fails and `assert_result` is True.
        """

        score = self.result[self.score_key]
        pass_state = (
                expected_score - allowed_uncertainty
                < score
                < expected_score + allowed_uncertainty
        )
        self.result.update(
            {
                "response": self.response,
                "expected_score": expected_score,
                f"{self.score_key}_result": 'pass' if pass_state else 'fail',
            }
        )

        return self.result

    def evaluate_against_golden_standards(
            self,
            golden_standards: List[str],
            scale_uncertainty: int = 1
    ):
        """
        Compares the evaluated score to the distribution of scores from a set of golden standard responses.

        Uses the mean ± scaled standard deviation of the golden scores as the acceptance range.

        Args:
            golden_standards (List[str]): A list of gold-standard responses for comparison.
            scale_uncertainty (int, optional): Scaling factor for the standard deviation. Defaults to 1.
            assert_result (bool, optional): Whether to raise an error on failure. Defaults to False.

        Returns:
            dict: A dictionary including golden stats, individual scores, and pass/fail result.

        Raises:
            AssertionError: If the result fails and `assert_result` is True.
        """

        current_score = self.result[self.score_key]
        golden_scores = []
        for golden_response in golden_standards:
            evaluator = self.evaluator_class()
            golden_scores.append(evaluator(response=golden_response)[self.score_key])
        score_mean = mean(golden_scores)
        score_uncertainty = stdev(golden_scores) * scale_uncertainty
        pass_state = (
                score_mean - score_uncertainty
                < current_score
                < score_mean + score_uncertainty
        )
        self.result.update(
            {
                "response": self.response,
                "golden_standard_responses": golden_standards,
                "golden_standard_scores": golden_scores,
                "mean_score": score_mean,
                "calculated_uncertainty": score_uncertainty,
                f"{self.score_key}_result": 'pass' if pass_state else 'fail',
            }
        )
        return self.result
