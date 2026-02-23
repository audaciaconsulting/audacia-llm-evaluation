from pathlib import Path

import pytest

from llm_eval.evaluators.llm_as_judge import (
    RunLlmAsJudgePassFailEvaluator,
    RunLlmAsJudgeScoreEvaluator,
)


def template_md(template_name: str):
    repo_root = Path(__file__).resolve().parents[2]
    template_path = repo_root / "llm_eval" / "prompt_templates" / f"{template_name}.md"
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


INPUTS_TARGET = """1. <input_1_key>
2. <input_2_key>
..."""

OBJECTIVES_TARGET = "<Clearly defined evaluation objective>"
METHODS_TARGET = "<Explicit instructions for how to compare or evaluate the inputs>"


def format_md(inputs, objective, methods, template_md):
    inputs_keys = list(inputs.keys())
    inputs_str = "\n".join(
        f"{i}. {key}"
        for i, key in enumerate(inputs_keys, start=1)
    )

    return (template_md.replace(INPUTS_TARGET, inputs_str)
            .replace(OBJECTIVES_TARGET, objective)
            .replace(METHODS_TARGET, methods))


@pytest.mark.parametrize(
    ("inputs", "objective", "methods", "template_name"),
    [
        (
                {"statement_1": "The Earth is an oblate spheroid",
                 "statement_2": "The Earth is bigger than the moon"
                 },
                "You are a fact checker, checking if all input statements are true",
                "Check if the statements are true, if all are true set llm_as_judge_result to 'pass', otherwise set it to 'fail'",
                "llm-as-judge-template"
        )
    ],
)
def test_llm_as_judge_pass(inputs: dict, objective: str, methods: str, template_name: str):
    template = template_md(template_name)
    prompt = format_md(inputs, objective, methods, template)
    evaluator = RunLlmAsJudgePassFailEvaluator(prompt=prompt, inputs=inputs)

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
    ("inputs", "objective", "methods", "template_name"),
    [
        (
                {"statement_1": "The Earth is an oblate spheroid",
                 "statement_2": "The Earth is bigger than the moon"
                 },
                "You are a fact checker, checking if all input statements are true",
                "Check if the statements are true, if all are true set llm_as_judge_result to 'pass', otherwise set it to 'fail'",
                "llm-as-judge-template"
        )
    ],
)
def test_llm_as_judge_fail_inputs_dict_prompt_mismatch(inputs: dict, objective: str, methods: str, template_name: str):
        with pytest.raises(ValueError):
            template = template_md(template_name)
            prompt = format_md(inputs, objective, methods, template)
            prompt = prompt.replace("statement_1", "claim_1").replace("statement_2", "claim_2")
            evaluator = RunLlmAsJudgePassFailEvaluator(prompt=prompt, inputs=inputs)
            evaluator.assert_result()


@pytest.mark.parametrize(
    ("inputs", "objective", "methods", "template_name"),
    [
        (
                {"statement_1": "The Earth is an oblate spheroid",
                 "statement_2": "The Earth is smaller than the moon"
                 },
                "You are a fact checker, checking if all input statements are true",
                "Check if the statements are true, if all are true set llm_as_judge_result to 'pass', otherwise set it to 'fail'",
                "llm-as-judge-template"
        )
    ],
)
def test_llm_as_judge_fail(inputs: dict, objective: str, methods: str, template_name: str):
    with pytest.raises(AssertionError):
        template = template_md(template_name)
        prompt = format_md(inputs, objective, methods, template)
        evaluator = RunLlmAsJudgePassFailEvaluator(prompt=prompt, inputs=inputs)
        evaluator.assert_result()


@pytest.mark.parametrize(
    ("inputs", "objective", "methods", "template_name", "threshold"),
    [
        (
                {
                    "statement_1": "Climate change is primarily driven by increased greenhouse gas emissions from human activities",
                    "statement_2": "Carbon dioxide levels have risen significantly since the Industrial Revolution",
                    "statement_3": "Many countries are investing in renewable energy, which completely eliminates carbon emissions"
                },
                "You are a fact checker, checking for truthfulness of input statements",
                "Give an overall score between 0 and 1 for truthfulness of all statements taken together and set it to llm_as_judge_result",
                "llm-as-judge-score-threshold-template",
                0.6
        )
    ],
)
def test_llm_as_judge_score_pass(inputs: dict, objective: str, methods: str, template_name: str, threshold: int):
    template = template_md(template_name)
    prompt = format_md(inputs, objective, methods, template)
    evaluator = RunLlmAsJudgeScoreEvaluator(prompt=prompt, threshold=threshold, inputs=inputs)

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
    ("inputs", "objective", "methods", "template_name", "threshold"),
    [
        (
                {
                    "statement_1": "Mars is the closest planet to the Sun",
                    "statement_2": "Mars has a dense, oxygen-rich atmosphere that supports human life",
                    "statement_3": "Mars is often studied by scientists because it may once have had liquid water on its surface."
                },
                "You are a fact checker, checking for truthfulness of input statements",
                "Give an overall score between 0 and 1 for truthfulness of all statements taken together and set it to llm_as_judge_result",
                "llm-as-judge-score-threshold-template",
                0.7
        )
    ],
)
def test_llm_as_judge_score_fail(inputs: dict, objective: str, methods: str, template_name: str, threshold: float):
    with pytest.raises(AssertionError):
        template = template_md(template_name)
        prompt = format_md(inputs, objective, methods, template)
        evaluator = RunLlmAsJudgeScoreEvaluator(prompt=prompt, threshold=threshold, inputs=inputs)
        evaluator.assert_result()
