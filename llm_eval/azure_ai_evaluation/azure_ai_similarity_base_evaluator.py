import logging
from abc import ABC, abstractmethod

from llm_eval.utils import format_dict_log

logger = logging.getLogger(__name__)


class BaseScoreEvaluator(ABC):
    """
    Base Evaluation Class for metric-based token-level evaluators.
    """

    def __init__(self, response: str, ground_truth: str, threshold: float):
        self.response = response
        self.ground_truth = ground_truth
        self.threshold = threshold

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0 and 1. Got {threshold}.")

    @abstractmethod
    def get_evaluator(self):
        """
        Returns an initialized evaluator object.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def get_result_key(self) -> str:
        """
        Returns the result key for pass/fail assertion.
        Must be implemented by subclasses.
        """
        pass

    async def __call__(self) -> dict:
        evaluator = self.get_evaluator()
        result = await evaluator._do_eval({
            "response": self.response,
            "ground_truth": self.ground_truth,
        })
        return result

    async def evaluate(self):
        result = await self()

        result.update({
            "response": self.response,
            "ground_truth": self.ground_truth,
        })

        logger.info(format_dict_log(dictionary=result))

        return result
        # assert result[self.get_result_key()] == 'pass'
