
from llm_eval.base_evaluators.custom_evaluators import SentimentEvaluator, AggregationStrategy
from llm_eval.base_evaluators.transformer_base_evaluator import (
    ExpectedScoreEvaluator,
    ReferenceScoresEvaluator,
)


class RunSentimentEvaluatorAgainstExpectedScore(ExpectedScoreEvaluator):
    """
    Evaluation runner for sentiment analysis in LLM responses.

    Computes an aggregate sentiment score using label weights and validates the result
    against known or gold-standard expectations.

    Args:
        response (str): The model-generated response to be evaluated.
        expected_score (float): The expected sentiment score for comparison.
        allowed_uncertainty (float, optional): Acceptable deviation from the expected
            score. Defaults to 0.05.
        aggregation_strategy (AggregationStrategy): How scores are aggregated across
            the response. Defaults to the full context.
    """

    def __init__(
        self, response: str, expected_score: float, allowed_uncertainty: float = 0.05, aggregation_strategy=AggregationStrategy.FULL_CONTEXT
    ):
        super().__init__(
            response=response,
            expected_score=expected_score,
            allowed_uncertainty=allowed_uncertainty,
            score_range=(-1.0, 1.0),
            score_key="sentiment",
            evaluator_class=SentimentEvaluator,
            assertion_fail_message="Evaluation failed: sentiment of response too different compared to expected score",
            aggregation_strategy=aggregation_strategy
        )


class RunSentimentEvaluatorAgainstReferences(ReferenceScoresEvaluator):
    """
    Sentiment Evaluation Runner Using Golden Standard Comparisons.

    This evaluator applies a transformer-based `SentimentEvaluator` to assess the sentiment of a given
    response and compares the resulting score against a set of golden standard responses.
    It determines pass/fail by checking if the response's sentiment score falls within an acceptable
    range defined by the statistical distribution (mean ± scaled standard deviation) of the
    golden responses.

    For this to work effectively include 10 or more reference responses, 3 is the absolute minimum

    Args:
        response (str): The model-generated response to be evaluated.
        references (list[str]): A list of reference responses with ideal sentiment.
        scale_uncertainty (int, optional): Scaling factor for standard deviation used to calculate
            the tolerance range. Defaults to 1.
    """

    def __init__(
        self, response: str, references: list[str], scale_uncertainty: int = 1
    ):
        super().__init__(
            response=response,
            references=references,
            scale_uncertainty=scale_uncertainty,
            score_key="sentiment",
            evaluator_class=SentimentEvaluator,
            assertion_fail_message="Evaluation failed: sentiment of response too different compared to golden standard responses",
        )
