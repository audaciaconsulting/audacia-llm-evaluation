import json
import logging
from typing import Any

from llm_eval.base_evaluators.format_base_evaluator import FormatBaseEvaluator

logger = logging.getLogger(__name__)


class RunCustomResponseEvaluator(FormatBaseEvaluator):
    """Evaluator for checking if a response matches an expected Python type."""

    def __init__(self, response: Any, expected_type: type):
        """
        Initialize the custom response evaluator.

        Args:
            response (Any): The model response to evaluate.
            expected_type (type): The expected Python type (e.g., dict, list, str).
        """
        self.expected_type = expected_type
        super().__init__(
            response=response,
            evaluator_name="custom_response",
            assertion_fail_message="Evaluation failed: output type of response not the expected format",
        )

    def _check(self) -> bool:
        return isinstance(self.response, self.expected_type)


class RunJsonResponseEvaluator(FormatBaseEvaluator):
    """Evaluator for checking if a response is valid JSON and is a dictionary."""

    def __init__(self, response: Any):
        """
        Initialize the JSON response evaluator.

        Args:
            response (Any): The response string to evaluate.
        """
        super().__init__(
            response=response,
            evaluator_name="json_response",
            assertion_fail_message="Evaluation failed: output is not a valid JSON format",
        )

    def _check(self) -> bool:
        try:
            return isinstance(json.loads(self.response), dict)
        except (json.JSONDecodeError, TypeError) as error:
            logger.info("JSON parsing failed: %s", error)
            return False
