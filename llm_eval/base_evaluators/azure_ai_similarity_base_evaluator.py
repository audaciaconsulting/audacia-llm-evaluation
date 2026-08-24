from llm_eval.base_evaluators.evaluator import AsyncEvaluator
from llm_eval.results import EvalResult


class BaseScoreEvaluator(AsyncEvaluator):
    """Scores a response against a reference with an azure-ai-evaluation scorer."""

    def __init__(
        self,
        response: str,
        ground_truth: str,
        threshold: float,
        result_key: str,
        evaluator: type,
        assertion_fail_message: str,
    ):
        """Initialize the score evaluator with comparison parameters.

        Args:
            response (str): Model-generated response to evaluate.
            ground_truth (str): Expected response to compare against.
            threshold (float): Minimum acceptable similarity score between 0 and 1.
            result_key (str): Key in the evaluation result indicating pass or fail.
            evaluator (type): Evaluator implementation providing `_do_eval`.
            assertion_fail_message (str): Message for assertion failures.

        Raises:
            ValueError: If `threshold` falls outside the inclusive [0, 1] range.
        """
        self.response = response
        self.ground_truth = ground_truth
        self.threshold = threshold
        self.result_key = result_key
        self.evaluator = evaluator
        self.assertion_fail_message = assertion_fail_message
        # e.g. "bleu_result" -> "bleu", the metric this evaluator reports.
        self.name = (
            result_key[: -len("_result")] if result_key.endswith("_result") else result_key
        )

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0 and 1. Got {threshold}.")

    async def _evaluate_async(self) -> EvalResult:
        scored = await self.evaluator._do_eval(
            {
                "response": self.response,
                "ground_truth": self.ground_truth,
            }
        )

        return EvalResult(
            name=self.name,
            passed=scored.get(self.result_key) == "pass",
            # ROUGE's key already ends in _score; the rest are "{name}_score".
            score=scored.get(f"{self.name}_score", scored.get(self.name)),
            threshold=self.threshold,
            inputs={"response": self.response, "ground_truth": self.ground_truth},
            raw={
                **scored,
                "response": self.response,
                "ground_truth": self.ground_truth,
            },
        )
