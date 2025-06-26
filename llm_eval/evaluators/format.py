import json
from typing import Any

from llm_eval.base_evaluators.format_base_evaluator import FormatBaseEvaluator


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
        super().__init__(response=response, evaluator_name="custom_response", assertion_fail_message="Evaluation failed: output type of response not the expected format")


    def evaluate(self):
        return self._format_result(isinstance(self.response, self.expected_type))


class RunJsonResponseEvaluator(FormatBaseEvaluator):
    """Evaluator for checking if a response is valid JSON and is a dictionary."""

    def __init__(self, response: Any):
        """
        Initialize the JSON response evaluator.

        Args:
            response (Any): The response string to evaluate.
        """
        super().__init__(response=response, evaluator_name="json_response", assertion_fail_message="Evaluation failed: output is not a valid JSON format")

    def evaluate(self):
        try:
            parsed = json.loads(self.response)
            is_valid = isinstance(parsed, dict)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"[Error] JSON parsing failed: {e}")
            is_valid = False

        return self._format_result(is_valid)
