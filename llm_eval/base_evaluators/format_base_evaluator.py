import logging
from typing import Any
from abc import ABC, abstractmethod

from llm_eval.tools.utils import format_dict_log

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class FormatBaseEvaluator(ABC):
    """Base class for evaluating the format of a model response."""

    def __init__(self, response: Any, evaluator_name: str):
        """
        Initialize the base format evaluator.

        Args:
            response (Any): The model response to evaluate.
            evaluator_name (str): Name of the specific evaluator subclass.
            assert_fail_message (str): Error message to use if the assertion fails.
            assert_result (bool): Whether to raise an assertion error if the result fails.
        """
        self.response = response
        self.evaluator_name = evaluator_name

    def __call__(self):
        result = self.evaluate()
        logger.info(format_dict_log(dictionary=result))
        return result

    def _format_result(self, result_flag: bool):
        return {
            "response": self.response,
            "format": type(self.response),
            f"{self.evaluator_name}_result": "pass" if result_flag else "fail",
        }

    def assert_result(self):
        result = self.evaluate()
        if result.get(f"{self.evaluator_name}_result") == "fail":
            message_type = (
                "as the response is in the incorrect format"
                if self.evaluator_name == "custom_response"
                else "as the response is not in a valid JSON format"
            )
            raise AssertionError(f"Evaluation failed {message_type}")

    @abstractmethod
    def evaluate(self):
        pass
