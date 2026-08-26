from ragas.dataset_schema import SingleTurnSample

from llm_eval.base_evaluators.evaluator import AsyncEvaluator
from llm_eval.results import EvalResult
from llm_eval.tools.utils import camel_to_snake

#: Inputs holding retrieved text, shortened when logged.
_CONTEXT_KEYS = ("retrieved_contexts", "reference_contexts")


class RagasBaseEvaluator(AsyncEvaluator):
    """Scores sample data with a Ragas metric."""

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

        Args:
            sample_data (dict): The inputs to score, as the metric's sample fields.
            threshold (float): Minimum passing score between 0 and 1. A bool selects
                binary scoring, where the score is rounded and 1 passes.
            ragas_metric (type): The Ragas metric class to use for scoring.
            assertion_fail_message (str): Message for assertion failures.
            ragas_metric_args (dict, optional): Arguments to pass to the metric, such
                as an llm or embedding model.

        Raises:
            ValueError: If a float `threshold` falls outside the inclusive [0, 1] range.
        """
        if ragas_metric_args is None:
            ragas_metric_args = {}

        self.sample_data = sample_data
        self.threshold = threshold
        self.ragas_metric = ragas_metric
        self.ragas_metric_args = ragas_metric_args
        self.name = camel_to_snake(self.ragas_metric.__name__)
        self.assertion_fail_message = assertion_fail_message

        if isinstance(self.threshold, float):
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(f"Threshold must be between 0 and 1. Got {threshold}.")

    async def _evaluate_async(self) -> EvalResult:
        """Scores the sample data and determines if it passes the threshold."""
        score = await self.ragas_metric(**self.ragas_metric_args).single_turn_ascore(
            sample=SingleTurnSample(**self.sample_data)
        )

        binary = isinstance(self.threshold, bool)
        passed = round(score) == 1 if binary else score >= self.threshold

        return EvalResult(
            name=self.name,
            passed=passed,
            score=score,
            # Binary metrics have no threshold; `raw` keeps the False it has
            # always reported.
            threshold=None if binary else self.threshold,
            inputs=self.sample_data,
            raw={
                **self.sample_data,
                self.name: score,
                f"{self.name}_threshold": self.threshold,
                f"{self.name}_result": "pass" if passed else "fail",
            },
        )

    def _log_payload(self, result: EvalResult) -> dict:
        """Shortens retrieved contexts, which can be whole source documents.

        Only the log is shortened. The result keeps them in full, because diagnosing
        a low score needs the text the score was derived from.
        """
        payload = dict(result.raw)

        for key in _CONTEXT_KEYS:
            if key in payload:
                payload[key] = [
                    text if len(text) <= 200 else f"{text[:100]}......{text[-100:]}"
                    for text in payload[key]
                ]

        return payload
