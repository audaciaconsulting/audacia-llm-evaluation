import logging
from llm_eval.tools.utils import format_dict_log

logger = logging.getLogger(__name__)


class BaseScoreEvaluator:
    def __init__(
        self,
        response: str,
        ground_truth: str,
        threshold: float,
        result_key: str,
        evaluator: type,
        assertion_fail_message: str,
    ):
        self.response = response
        self.ground_truth = ground_truth
        self.threshold = threshold
        self.result_key = result_key
        self.evaluator = evaluator
        self.assertion_fail_message = assertion_fail_message

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0 and 1. Got {threshold}.")

    async def __call__(self) -> dict:
        result = await self.evaluator._do_eval(
            {
                "response": self.response,
                "ground_truth": self.ground_truth,
            }
        )
        logger.info(format_dict_log(dictionary=result))
        return result

    async def assert_result(self):
        result = await self()
        if result.get(f"{self.result_key}") == "fail":
            raise AssertionError(self.assertion_fail_message)
