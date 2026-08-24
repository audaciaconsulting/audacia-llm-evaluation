from abc import abstractmethod
from statistics import mean, stdev
from typing import List, Tuple

from llm_eval.base_evaluators.custom_evaluators import AggregationStrategy
from llm_eval.base_evaluators.evaluator import Evaluator
from llm_eval.results import EvalResult


class TransformerRunEvaluator(Evaluator):
    """
    Base class for running transformer-based evaluation on a response and validating
    it against expectations or golden standards.

    This class defines a reusable evaluation interface for subclasses like sentiment,
    bias, or toxicity evaluators that wrap specific Transformer models and scoring
    logic.

    Attributes:
        response (str): The textual response to be evaluated.

    Subclasses supply `evaluator_class` (the model wrapper), `score_key` (which key in
    its output holds the score), and `_judge` (how that score is decided on).
    """

    def __init__(
        self,
        response: str,
        score_key: str,
        evaluator_class: type,
        assertion_fail_message: str,
        aggregation_strategy=AggregationStrategy.FULL_CONTEXT,
    ):
        """Initialize the transformer evaluator.

        Args:
            response (str): The textual response to be evaluated.
            score_key (str): Key in the model's output holding the score.
            evaluator_class (type): The evaluator class wrapping the model.
            assertion_fail_message (str): Error message to use if the assertion fails.
            aggregation_strategy (AggregationStrategy): How scores are aggregated
                across the response. Defaults to the full context.
        """
        self.response = response
        self.score_key = score_key
        self.name = score_key
        self.evaluator_class = evaluator_class
        self.assertion_fail_message = assertion_fail_message
        self.aggregation_strategy = aggregation_strategy

    @abstractmethod
    def _judge(self, score: float) -> Tuple[bool, dict]:
        """Decide on `score`.

        Args:
            score (float): The score the model gave the response.

        Returns:
            Tuple[bool, dict]: Whether it passed, and the fields explaining why, which
                are added to the result's `raw`.
        """

    def _score(self, response: str) -> dict:
        """Runs the model over `response`."""
        evaluator = self.evaluator_class(self.aggregation_strategy)
        return evaluator(response=response)

    def _evaluate(self) -> EvalResult:
        """
        Runs the evaluation on the response using the configured evaluator class.

        Returns:
            EvalResult: The outcome, with the model's scores in `raw`.
        """
        scores = self._score(self.response)
        score = scores[self.score_key]
        passed, detail = self._judge(score)

        return EvalResult(
            name=self.name,
            passed=passed,
            score=score,
            inputs={"response": self.response},
            raw={
                **scores,
                **detail,
                "response": self.response,
                f"{self.score_key}_result": "pass" if passed else "fail",
            },
        )


class ExpectedScoreEvaluator(TransformerRunEvaluator):
    """
    Compares the evaluated score to an expected score within an uncertainty margin.

    Attributes:
        response (str): The textual response to be evaluated.
    """

    def __init__(
        self,
        response: str,
        expected_score: float,
        score_key: str,
        evaluator_class: type,
        assertion_fail_message: str,
        allowed_uncertainty: float = 0.05,
        aggregation_strategy=AggregationStrategy.FULL_CONTEXT,
    ):
        """Initialize the expected-score evaluator.

        Args:
            response (str): The textual response to be evaluated.
            expected_score (float): The target score the response should be close to.
            score_key (str): Key in the model's output holding the score.
            evaluator_class (type): The evaluator class wrapping the model.
            assertion_fail_message (str): Error message to use if the assertion fails.
            allowed_uncertainty (float, optional): Acceptable deviation from the
                expected score. Defaults to 0.05.
            aggregation_strategy (AggregationStrategy): How scores are aggregated
                across the response. Defaults to the full context.
        """
        self.expected_score = expected_score
        self.allowed_uncertainty = allowed_uncertainty
        super().__init__(
            response=response,
            score_key=score_key,
            evaluator_class=evaluator_class,
            assertion_fail_message=assertion_fail_message,
            aggregation_strategy=aggregation_strategy,
        )

    def _judge(self, score: float) -> Tuple[bool, dict]:
        passed = (
            self.expected_score - self.allowed_uncertainty
            < score
            < self.expected_score + self.allowed_uncertainty
        )
        return passed, {"expected_score": self.expected_score}


class ReferenceScoresEvaluator(TransformerRunEvaluator):
    """
    Compares the evaluated score to the spread of golden standard scores.

    Uses the mean of the reference scores plus or minus their standard deviation,
    scaled by `scale_uncertainty`, as the acceptance range. Include ten or more
    references for this to work effectively; three is the absolute minimum.

    Attributes:
        response (str): The textual response to be evaluated.
    """

    def __init__(
        self,
        response: str,
        references: List[str],
        score_key: str,
        evaluator_class: type,
        assertion_fail_message: str,
        scale_uncertainty: int = 1,
        aggregation_strategy=AggregationStrategy.FULL_CONTEXT,
    ):
        """Initialize the reference-scores evaluator.

        Args:
            response (str): The textual response to be evaluated.
            references (List[str]): Gold-standard responses for comparison.
            score_key (str): Key in the model's output holding the score.
            evaluator_class (type): The evaluator class wrapping the model.
            assertion_fail_message (str): Error message to use if the assertion fails.
            scale_uncertainty (int, optional): Scaling factor for the standard
                deviation. Defaults to 1.
            aggregation_strategy (AggregationStrategy): How scores are aggregated
                across the response. Defaults to the full context.
        """
        self.references = references
        self.scale_uncertainty = scale_uncertainty
        super().__init__(
            response=response,
            score_key=score_key,
            evaluator_class=evaluator_class,
            assertion_fail_message=assertion_fail_message,
            aggregation_strategy=aggregation_strategy,
        )

    def _judge(self, score: float) -> Tuple[bool, dict]:
        reference_scores = [
            self._score(reference)[self.score_key] for reference in self.references
        ]
        score_mean = mean(reference_scores)
        uncertainty = stdev(reference_scores) * self.scale_uncertainty
        passed = score_mean - uncertainty < score < score_mean + uncertainty

        return passed, {
            "references": self.references,
            "reference_scores": reference_scores,
            "mean_score": score_mean,
            "calculated_uncertainty": uncertainty,
        }
