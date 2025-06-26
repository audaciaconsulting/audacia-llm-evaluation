import pytest
from typing import Any

from llm_eval.evaluators.format import (
    RunCustomResponseEvaluator,
    RunJsonResponseEvaluator,
)


@pytest.mark.parametrize(
    "response, expected_type, expected_result",
    [
        ("some string", str, "pass"),
        ({"key": "value"}, dict, "pass"),
        ([1, 2, 3], list, "pass"),
        (123, str, "fail"),
    ],
)
def test_evaluate_custom_response(
    response: Any, expected_type: type, expected_result: bool
):
    eval = RunCustomResponseEvaluator(response, expected_type)
    result = eval()
    assert result["custom_response_result"] == expected_result
    assert all(
        key in result for key in ["response", "format", "custom_response_result"]
    )


def test_evaluate_custom_response_assert_passes():
    RunCustomResponseEvaluator(
        response={"key": "value"}, expected_type=dict
    ).assert_result()

def test_evaluate_custom_response_assert_fails():
    with pytest.raises(
        AssertionError,
        match="Evaluation failed: output type of response not the expected format",
    ):
        RunCustomResponseEvaluator(
            response={"not a dict"}, expected_type=dict
        ).assert_result()


@pytest.mark.parametrize(
    "json_str, expected_result",
    [
        ('{"a": 1}', "pass"),
        ("[1, 2, 3]", "fail"),
        ('"just a string"', "fail"),
        ("invalid json", "fail"),
        (1234, "fail"),  # Not a string at all
    ],
)
def test_evaluate_json_response(json_str: Any, expected_result: bool):
    eval = RunJsonResponseEvaluator(response=json_str)
    result = eval()
    assert result["json_response_result"] == expected_result
    assert all(
        key in result for key in ["response", "format", "json_response_result"]
    )


def test_evaluate_json_response_assert_passes():
    RunJsonResponseEvaluator(response='{"valid": true}').assert_result()


def test_evaluate_json_response_assert_fails_on_invalid_json():
    with pytest.raises(
        AssertionError,
        match="Evaluation failed: output is not a valid JSON format",
    ):
        RunJsonResponseEvaluator(response='{"missing": "value"').assert_result()


def test_evaluate_json_response_assert_fails_on_non_dict():
    with pytest.raises(
        AssertionError,
        match="Evaluation failed: output is not a valid JSON format",
    ):
        RunJsonResponseEvaluator(response='["not", "a", "dict"]').assert_result()
