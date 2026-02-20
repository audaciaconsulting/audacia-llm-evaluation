import pytest

from llm_eval.evaluators.llm_as_judge import RunLlmAsJudgePassFailEvaluator


@pytest.mark.parametrize(
    "prompt",
    [
        (
            """
            Fact Checker Eval Prompt
            
            You are a fact checker. If the statement is true the **result** is "pass" else **result** is "false"

            Output in JSON
            ```JSON
            {
              "llm_as_judge_result":  "<score>",
              "failures_list": "[<failures>]"
            }
            ```
            Statement:
            The earth is an oblate spheroid
            """,
        )
    ],
)
def test_llm_as_judge_pass(
    prompt: str,
):
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
    
                You are a fact checker. If the statement is true the **result** is "pass" else **result** is "false"
    
                Output in JSON
                ```JSON
                {
                  "llm_as_judge_result":  "<score>",
                  "failures_list": "[<failures>]"
                }
                ```
                Statement:
                The earth is flat
                """,
        )
    ],
)
def test_llm_as_judge_fail(
        prompt: str,
):
    with pytest.raises(AssertionError):
        evaluator = RunLlmAsJudgePassFailEvaluator(prompt=prompt)
        evaluator.assert_result()
