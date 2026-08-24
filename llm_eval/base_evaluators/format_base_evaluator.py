from abc import abstractmethod
from typing import Any

from llm_eval.base_evaluators.evaluator import Evaluator
from llm_eval.results import EvalResult


class FormatBaseEvaluator(Evaluator):
    """Base class for evaluating the format of a model response."""

    def __init__(self, response: Any, evaluator_name: str, assertion_fail_message: str):
        """
        Initialize the base format evaluator.

        Args:
            response (Any): The model response to evaluate.
            evaluator_name (str): Name of the specific evaluator subclass.
            assertion_fail_message (str): Error message to use if the assertion fails.
        """
        self.response = response
        self.evaluator_name = evaluator_name
        self.name = evaluator_name
        self.assertion_fail_message = assertion_fail_message

    @abstractmethod
    def _check(self) -> bool:
        """Whether `self.response` is in the expected format."""

    def _evaluate(self) -> EvalResult:
        passed = self._check()

        return EvalResult(
            name=self.name,
            passed=passed,
            inputs={"response": self.response},
            raw={
                "response": self.response,
                "format": type(self.response),
                f"{self.evaluator_name}_result": "pass" if passed else "fail",
            },
        )
