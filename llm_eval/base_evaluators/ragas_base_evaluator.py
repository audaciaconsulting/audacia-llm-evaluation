from ragas.dataset_schema import SingleTurnSample
import logging
from llm_eval.tools.utils import format_dict_log, camel_to_snake

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class RagasBaseEvaluator:
    def __init__(
        self,
        sample_data: dict,
        threshold: float,
        ragas_metric: type,
        assertion_fail_message: str,
        ragas_metric_args: dict = None,
    ):
        """
        Initializes the evaluator.

        :param response: The model's response to evaluate.
        :param reference: The reference answer to compare against.
        :param threshold: The threshold value for evaluation.
        :param ragas_metric: The Ragas metric class to use for scoring.
        :param ragas_metric_args: Arguments to pass to the metric (optional).
        """
        if ragas_metric_args is None:
            ragas_metric_args = {}

        self.sample_data = sample_data
        self.threshold = threshold
        self.ragas_metric = ragas_metric
        self.ragas_metric_args = ragas_metric_args
        self.metric_name = camel_to_snake(self.ragas_metric.__name__)
        self.metric_name_result = f"{self.metric_name}_result"
        self.assertion_fail_message = assertion_fail_message

        if isinstance(self.threshold, float):
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(f"Threshold must be between 0 and 1. Got {threshold}.")

    async def __call__(self) -> dict:
        """
        Scores the response and determines if it passes the threshold.
        """
        sample = SingleTurnSample(**self.sample_data)
        score = await self.ragas_metric(**self.ragas_metric_args).single_turn_ascore(
            sample=sample
        )

        if isinstance(self.threshold, bool):
            pass_eval = "pass" if round(score) == 1 else "fail"
        else:
            pass_eval = "pass" if score >= self.threshold else "fail"

        results = {
            **self.sample_data,
            self.metric_name: score,
            f"{self.metric_name}_threshold": self.threshold,
            self.metric_name_result: pass_eval,
        }

        logger.info(format_dict_log(dictionary=results))

        return results

    async def assert_result(self):
        result = await self()
        if result.get(f"{self.metric_name_result}") == "fail":
            raise AssertionError(self.assertion_fail_message)
