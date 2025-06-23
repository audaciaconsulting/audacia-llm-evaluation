from llm_eval.custom_evaluator_tools import FormatEvaluator
import logging
from typing import Any
import json

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class RunFormatEvaluator:
    def __init__(self, response: Any):
        self.response = response

    def __call__(self):
        evaluator = FormatEvaluator()
        obj_format = evaluator(response = self.response)
        return obj_format
    
    def evaluate_custom_response(self, expected_format:Any):
        return isinstance(self.response, expected_format)
    
    def evaluate_json_response(self):
        try:
            response = json.loads(self.response)
            pass_state = isinstance(response, dict)
        except:
            pass_state = False
        return pass_state

