import pytest

from llm_eval.evaluators.llm_as_judge import (
    RunLlmAsJudgePassFailEvaluator,
    RunLlmAsJudgeScoreEvaluator,
)


@pytest.mark.parametrize(
    "prompt",
    [
        (
            """
            Fact Checker Eval Prompt

            You are a fact checker. If the statement is true set "llm_as_judge_result" to "pass",
            otherwise set it to "fail".

            Output strictly as JSON matching:
            {
              "llm_as_judge_result": "pass|fail",
              "failures_list": ["<failure_1>", "<failure_2>"]
            }

            Statement:
            The earth is an oblate spheroid.
            """,
        )
    ],
)
def test_llm_as_judge_pass(prompt: str):
    evaluator = RunLlmAsJudgePassFailEvaluator(prompt=prompt)
    evaluator.assert_result()

    result = evaluator()
    assert all(
        key in result
        for key in [
            "llm_as_judge_result",
            "failures_list",
            "prompt_trunc",
        ]
    )


@pytest.mark.parametrize(
    "prompt",
    [
        (
            """
            Fact Checker Eval Prompt

            You are a fact checker. If the statement is true set "llm_as_judge_result" to "pass",
            otherwise set it to "fail".

            Output strictly as JSON matching:
            {
              "llm_as_judge_result": "pass|fail",
              "failures_list": ["<failure_1>", "<failure_2>"]
            }

            Statement:
            The earth is flat.
            """,
        )
    ],
)
def test_llm_as_judge_fail(prompt: str):
    with pytest.raises(AssertionError):
        evaluator = RunLlmAsJudgePassFailEvaluator(prompt=prompt)
        evaluator.assert_result()


@pytest.mark.parametrize(
    "prompt, threshold",
    [
        (
            """
            Scoring Eval Prompt

            You are a fact checker. Return a score between 0.0 and 1.0 where 1.0 is fully correct
            and 0.0 is fully incorrect.

            Output strictly as JSON matching:
            {
              "llm_as_judge_score": 0.0,
              "failures_list": ["<failure_1>", "<failure_2>"]
            }

            Statement:
            Climate change is primarily driven by increased greenhouse gas emissions from human activities, and carbon dioxide levels have risen significantly since the Industrial Revolution. As a result, many countries are investing in renewable energy, which completely eliminates carbon emissions.
            """,
            0.7,
        )
    ],
)
def test_llm_as_judge_score_pass(prompt: str, threshold: float):
    evaluator = RunLlmAsJudgeScoreEvaluator(prompt=prompt, threshold=threshold)
    evaluator.assert_result()

    result = evaluator()
    assert all(
        key in result
        for key in [
            "llm_as_judge_score",
            "llm_as_judge_result",
            "threshold",
            "failures_list",
            "prompt_trunc",
        ]
    )


@pytest.mark.parametrize(
    "prompt, threshold",
    [
        (
            """
            Scoring Eval Prompt

            You are a fact checker. Return a score between 0.0 and 1.0 where 1.0 is fully correct
            and 0.0 is fully incorrect.

            Output strictly as JSON matching:
            {
              "llm_as_judge_score": 0.0,
              "failures_list": ["<failure_1>", "<failure_2>"]
            }

            Statement:
            Mars is the closest planet to the Sun, and it has a dense, oxygen-rich atmosphere that supports human life. It is often studied by scientists because it may once have had liquid water on its surface.
            """,
            0.7,
        )
    ],
)
def test_llm_as_judge_score_fail(prompt: str, threshold: float):
    with pytest.raises(AssertionError):
        evaluator = RunLlmAsJudgeScoreEvaluator(prompt=prompt, threshold=threshold)
        evaluator.assert_result()
