from llm_eval.custom_evaluator_tools import FormatEvaluator
from llm_eval.utils import format_dict_log
import logging
from typing import Any
import json

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class RunFormatEvaluator:
    """Evaluates the format of a model response for correctness and structure.

    This class provides utilities to:
    - Identify the type of a response using `FormatEvaluator`
    - Log the response format
    - Check if the response matches a specified type
    - Validate whether the response is a well-formed JSON dictionary

    Attributes:
        response (Any): The response object to evaluate.

    Methods:
        __call__(): Runs the FormatEvaluator and logs the resulting format.
        evaluate_custom_response(expected_format): Checks if response is an instance of the expected format.
        evaluate_json_response(): Attempts to parse the response as JSON and verifies it's a dictionary.

    Example:
        evaluator = RunFormatEvaluator(response='{"key": "value"}')
        evaluator()
        evaluator.evaluate_custom_response(dict)
        evaluator.evaluate_json_response()
    """
    def __init__(self, response: Any):
        """Initializes the RunFormatEvaluator.

        Args:
            response (Any): The response to evaluate.
        """
        self.response = response

    def __call__(self):
        """Evaluates and logs the type of the response using FormatEvaluator.

        Returns:
            dict: A dictionary with a single key `'format'` indicating the type of the response.
        """
        evaluator = FormatEvaluator()
        obj_format = evaluator(response = self.response)
        logger.info(format_dict_log(dictionary=obj_format))
        return obj_format
    
    def evaluate_custom_response(self, expected_format:Any, assert_result: bool = False):
        """Checks if the response is an instance of the expected format.

        Args:
            expected_format (Any): The expected Python type (e.g., `dict`, `list`, `str`).
            assert_result (bool, optional): If True, raises an AssertionError when the response 
                does not match the expected format. Defaults to False.

        Returns:
            bool: True if the response matches the expected type; False otherwise.
                If `assert_result` is True, the method raises an error instead of returning.
        
        Raises:
            AssertionError: If `assert_result` is True and the response does not match the expected format.
        """
        if assert_result:
            assert isinstance(self.response, expected_format), "The response is in the incorrect format"
        else:
            return isinstance(self.response, expected_format)
    
    def evaluate_json_response(self, assert_result: bool = False):
        """Attempts to parse the response as JSON and checks if it results in a dictionary.

        Args:
            assert_result (bool, optional): If True, raises an AssertionError if the response
                is not valid JSON or not a dictionary. Defaults to False.

        Returns:
            bool: True if the response is a valid JSON object and parsed as a dictionary; False otherwise.
                If `assert_result` is True, the method raises an error instead of returning.

        Raises:
            AssertionError: If `assert_result` is True and the response is not valid JSON
                or not a dictionary.
        """
        try:
            response = json.loads(self.response)
            pass_state = isinstance(response, dict)
        except json.JSONDecodeError as e:
            print(f"[JSONDecodeError] Invalid JSON: {e}")
            pass_state = False
        except TypeError as e:
            print(f"[TypeError] Response must be a string, bytes or bytearray: {e}")
            pass_state = False
        if assert_result:
            assert pass_state, "The response is not in a valid JSON format"
        else:
            return pass_state
    


