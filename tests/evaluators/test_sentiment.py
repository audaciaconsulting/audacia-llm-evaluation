import pytest

from llm_eval.evaluators.sentiment import (
    RunSentimentEvaluatorAgainstExpectedScore,
    RunSentimentEvaluatorAgainstReferences,
)

GOLDEN_STANDARDS = [
    "I can't believe how amazing this is — you've truly outdone yourselves!",
    "This is absolutely incredible, I feel like my life just changed for the better!",
    "What a phenomenal product — you've brought so much joy into my day!",
    "I'm completely blown away, this is hands-down the best thing I've ever owned!",
    "Thank you for creating something so wonderful — it genuinely made me emotional.",
    "This exceeded every expectation I had — I feel inspired and energized!",
    "Wow, just wow. You've made a lifelong fan out of me with this!",
    "I didn't know something could be this perfect — you've brightened my entire week!",
    "Unbelievably good — I wish I could give this ten stars!",
    "Pure brilliance — you’ve delivered something truly special and unforgettable.",
]


@pytest.fixture(
    params=[
        (
            "Oh wow, this is the best product I've ever received! You have made live worth living now!",
            0.8,
        ),
        ("This is a really cool product, good job", 0.4),
        ("This is a product", 0),
        ("This product isn't very good, try harder", -0.4),
        (
            "This is the worst product I've ever seen, the fact that you would even consider presenting this rubbish it to me is insulting",
            -0.8,
        ),
    ]
)
def simple_sentiment_case_(request):
    return request.param


@pytest.fixture(
    params=[
        (
            "Oh wow, this is the best product I've ever received! You have made live worth living now!",
            -0.8,
        ),
        ("This is a really cool product, good job", -0.4),
        ("This is a product", 1),
        ("This product isn't very good, try harder", 0.4),
        (
            "This is the worst product I've ever seen, the fact that you would even consider presenting this rubbish it to me is insulting",
            0.8,
        ),
    ]
)
def failure_sentiment_case_(request):
    return request.param


def test_expected_sentiment_score(simple_sentiment_case_):
    response_text, expected_score = simple_sentiment_case_

    result = RunSentimentEvaluatorAgainstExpectedScore(
        response=response_text, expected_score=expected_score
    )()

    assert "sentiment" in result
    assert expected_score - 0.2 <= result["sentiment"] <= expected_score + 0.2


def test_expected_sentiment_score_using_assert_method(failure_sentiment_case_):
    response_text, expected_score = failure_sentiment_case_

    evaluator = RunSentimentEvaluatorAgainstExpectedScore(
        response=response_text, expected_score=expected_score
    )

    with pytest.raises(
        AssertionError,
        match="Evaluation failed: sentiment of response too different compared to expected score",
    ):
        evaluator.assert_result()


def test_evaluate_sentiment_against_known_score(simple_sentiment_case_):
    response_text, expected_score = simple_sentiment_case_
    result = RunSentimentEvaluatorAgainstExpectedScore(
        response=response_text, expected_score=expected_score, allowed_uncertainty=0.2
    )()

    assert all(
        key in result
        for key in ["sentiment", "response", "expected_score", "sentiment_result"]
    )
    assert result["sentiment_result"] == "pass"


def test_evaluate_sentiment_against_golden_standards():
    response_text = "Oh wow, this is the best product I've ever received! You have made live worth living now!"

    result = RunSentimentEvaluatorAgainstReferences(response=response_text, references=GOLDEN_STANDARDS,
                                                    scale_uncertainty=3)()  # Scale to 3 standard deviations - acceptable

    assert all(
        key in result
        for key in [
            "sentiment",
            "response",
            "references",
            "reference_scores",
            "mean_score",
            "calculated_uncertainty",
            "sentiment_result",
        ]
    )
    assert result["references"] == GOLDEN_STANDARDS
    assert result["sentiment_result"] == "pass"
