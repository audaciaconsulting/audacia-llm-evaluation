import pytest
from typing import Any

from llm_eval.format_evaluator import RunFormatEvaluator


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
    evaluator = RunFormatEvaluator(response)
    assert evaluator.evaluate_custom_response(expected_type)['custom_response_result'] == expected_result


def test_evaluate_custom_response_assert_passes():
    evaluator = RunFormatEvaluator({"key": "value"})
    # Should not raise
    evaluator.evaluate_custom_response(dict, assert_result=True)


def test_evaluate_custom_response_assert_fails():
    evaluator = RunFormatEvaluator("not a dict")
    with pytest.raises(AssertionError, match="The response is in the incorrect format"):
        evaluator.evaluate_custom_response(dict, assert_result=True)


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
    evaluator = RunFormatEvaluator(json_str)
    assert evaluator.evaluate_json_response()['json_response_result'] == expected_result


def test_evaluate_json_response_assert_passes():
    evaluator = RunFormatEvaluator('{"valid": true}')
    # Should not raise
    evaluator.evaluate_json_response(assert_result=True)


def test_evaluate_json_response_assert_fails_on_invalid_json():
    evaluator = RunFormatEvaluator('{"missing": "value"')  # malformed JSON
    with pytest.raises(AssertionError, match="The response is not in a valid JSON format"):
        evaluator.evaluate_json_response(assert_result=True)


def test_evaluate_json_response_assert_fails_on_non_dict():
    evaluator = RunFormatEvaluator('["not", "a", "dict"]')
    with pytest.raises(AssertionError, match="The response is not in a valid JSON format"):
        evaluator.evaluate_json_response(assert_result=True)

