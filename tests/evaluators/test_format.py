import pytest
from typing import Any

from evaluators.format import RunCustomResponseEvaluator, RunJsonResponseEvaluator


@pytest.mark.parametrize(
    "response, expected_type, expected_result",
    [
        ("some string", str, 'pass'),
        ({"key": "value"}, dict, 'pass'),
        ([1, 2, 3], list, 'pass'),
        (123, str, 'fail'),
    ]
)
def test_evaluate_custom_response(response: Any, expected_type: type, expected_result: bool):
    eval = RunCustomResponseEvaluator(response, expected_type)
    result = eval()
    assert result['custom_response_result'] == expected_result


def test_evaluate_custom_response_assert_passes():
    eval = RunCustomResponseEvaluator(response={"key": "value"}, expected_type=dict, assert_result=True)
    eval()


def test_evaluate_custom_response_assert_fails():
    eval = RunCustomResponseEvaluator(response={"not a dict"}, expected_type=dict, assert_result=True)

    with pytest.raises(AssertionError, match="The response is in the incorrect format"):
        eval()

@pytest.mark.parametrize(
    "json_str, expected_result",
    [
        ('{"a": 1}', 'pass'),
        ('[1, 2, 3]', 'fail'),
        ('"just a string"', 'fail'),
        ('invalid json', 'fail'),
        (1234, 'fail'),  # Not a string at all
    ]
)
def test_evaluate_json_response(json_str: Any, expected_result: bool):
    eval = RunJsonResponseEvaluator(response=json_str)
    assert eval()['json_response_result'] == expected_result


def test_evaluate_json_response_assert_passes():
    eval = RunJsonResponseEvaluator(response='{"valid": true}', assert_result=True)
    eval()


def test_evaluate_json_response_assert_fails_on_invalid_json():
    eval = RunJsonResponseEvaluator(response='{"missing": "value"', assert_result=True)

    with pytest.raises(AssertionError, match="The response is not in a valid JSON format"):
        eval()


def test_evaluate_json_response_assert_fails_on_non_dict():
    eval = RunJsonResponseEvaluator(response='["not", "a", "dict"]', assert_result=True)

    with pytest.raises(AssertionError, match="The response is not in a valid JSON format"):
        eval()
