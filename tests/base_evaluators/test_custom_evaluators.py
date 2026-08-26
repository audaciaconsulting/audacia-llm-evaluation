"""Empty-input handling in the transformer-backed evaluators.

The tokenizer and classification pipeline are stubbed out: what is under test is the
guard in front of the model, not the model.
"""

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("transformers", reason="requires the 'local-models' extra")

from llm_eval.base_evaluators import custom_evaluators
from llm_eval.base_evaluators.custom_evaluators import (
    AggregationStrategy,
    BiasEvaluator,
    SentimentEvaluator,
    ToxicityEvaluator,
)
from llm_eval.evaluators.bias import RunBiasEvaluatorAgainstExpectedScore


@pytest.fixture
def classifier():
    """Stands in for the tokenizer and the classification pipeline."""
    with (
        patch.object(custom_evaluators, "AutoTokenizer") as tokenizer,
        patch.object(custom_evaluators, "pipeline") as pipeline,
    ):
        tokenizer.from_pretrained.return_value = MagicMock()
        yield pipeline.return_value


@pytest.mark.parametrize(
    "build", [BiasEvaluator, ToxicityEvaluator, SentimentEvaluator]
)
@pytest.mark.parametrize("strategy", list(AggregationStrategy))
def test_a_response_with_no_text_is_rejected(classifier, build, strategy):
    """It used to be a bare KeyError — or, aggregating, a silent score of 0.0."""
    evaluator = build(aggregation_strategy=strategy)

    with pytest.raises(ValueError, match="no text to score"):
        evaluator(response="   ")

    classifier.assert_not_called()


@pytest.mark.parametrize("response", ["", "   ", "\n\t"])
def test_what_counts_as_no_text(classifier, response):
    """Empty or whitespace-only: what leaves `_split_sentences` with nothing."""
    with pytest.raises(ValueError, match="no text to score"):
        BiasEvaluator()(response=response)


@pytest.mark.parametrize("build", [BiasEvaluator, SentimentEvaluator])
def test_no_classification_results_is_rejected(classifier, build):
    """Both paths through `_extract_score_from_results`: label lookup and weights."""
    with pytest.raises(ValueError, match="no classification results"):
        build()._extract_score_from_results([])


def test_the_error_reaches_the_consumer(classifier):
    """Nothing on the way out turns it back into a score."""
    evaluator = RunBiasEvaluatorAgainstExpectedScore(
        response="   ", expected_score=0.5
    )

    with pytest.raises(ValueError, match="no text to score"):
        evaluator()
