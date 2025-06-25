import json
import logging
from typing import Any
from abc import ABC, abstractmethod

from llm_eval.tools.utils import format_dict_log

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class BaseFormatEvaluator(ABC):
    """Base class for evaluating the format of a model response."""

    def __init__(self, response: Any, evaluator_name: str, assert_fail_message: str, assert_result: bool = False):
        self.response = response
        self.evaluator_name = evaluator_name
        self.assert_fail_message = assert_fail_message
        self.assert_result = assert_result

    def __call__(self):
        result = self.evaluate()
        logger.info(format_dict_log(dictionary=result))

        if self.assert_result:
            assert result[f'{self.evaluator_name}_result'] == 'pass', self.assert_fail_message

        return result

    def _format_result(self, result_flag: bool):
        return {
            "response": self.response,
            "format": type(self.response),
            f"{self.evaluator_name}_result": "pass" if result_flag else "fail"
        }

    @abstractmethod
    def evaluate(self):
        pass


class RunCustomResponseEvaluator(BaseFormatEvaluator):
    """Evaluator for checking if a response matches an expected Python type."""

    def __init__(self, response: Any, expected_type: type, assert_result: bool = False):
        self.expected_type = expected_type
        super().__init__(
            response=response,
            evaluator_name='custom_response',
            assert_fail_message="The response is in the incorrect format",
            assert_result=assert_result
        )

    def evaluate(self):
        return self._format_result(isinstance(self.response, self.expected_type))


class RunJsonResponseEvaluator(BaseFormatEvaluator):
    """Evaluator for checking if a response is valid JSON and is a dictionary."""

    def __init__(self, response: Any, assert_result: bool = False):
        super().__init__(
            response=response,
            evaluator_name='json_response',
            assert_fail_message="The response is not in a valid JSON format",
            assert_result=assert_result
        )

    def evaluate(self):
        try:
            parsed = json.loads(self.response)
            is_valid = isinstance(parsed, dict)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"[Error] JSON parsing failed: {e}")
            is_valid = False

        return self._format_result(is_valid)


