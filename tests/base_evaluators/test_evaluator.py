"""Tests for the interface every evaluator shares.

Uses evaluators that need no model or credentials, so this runs anywhere. One is
async internally (ragas string presence), one is not (JSON format) — the point being
that a consumer cannot tell them apart from the outside.
"""

import asyncio

import pytest

from llm_eval.evaluators.format import RunJsonResponseEvaluator
from llm_eval.evaluators.similarity import RunStringPresenceEvaluator
from llm_eval.results import EvalResult

SIGN_OFF = "Kind regards, the support team"


def _async_backed(reference=SIGN_OFF):
    """An evaluator that awaits a Ragas metric internally."""
    return RunStringPresenceEvaluator(response=SIGN_OFF, reference=reference)


def _sync_backed(response='{"a": 1}'):
    """An evaluator with no awaiting anywhere."""
    return RunJsonResponseEvaluator(response=response)


# Factories, not instances: an instance built in the decorator is shared across
# tests and would need credentials at collection time.
@pytest.mark.parametrize("build", [_async_backed, _sync_backed])
def test_every_evaluator_returns_an_eval_result(build):
    result = build()()

    assert isinstance(result, EvalResult)
    assert result.passed is True
    assert result.result == "pass"
    assert result.name


@pytest.mark.parametrize("build", [_async_backed, _sync_backed])
def test_evaluate_and_call_agree(build):
    evaluator = build()

    assert evaluator.evaluate().passed == evaluator().passed


def test_raw_keeps_the_flat_metric_dict():
    """Consumers read `{metric}_result` keys; that convention must not change."""
    result = _async_backed()()

    assert result.raw["string_presence_result"] == "pass"
    assert result.raw["string_presence"] == 1.0
    assert "string_presence_threshold" in result.raw
    assert result.raw["response"] == SIGN_OFF


def test_binary_metric_has_no_threshold():
    """Binary metrics round the score, so there is nothing to compare against."""
    result = _async_backed()()

    assert result.threshold is None
    # `raw` keeps the False these have always reported
    assert result.raw["string_presence_threshold"] is False


def test_assert_result_returns_the_result_when_it_passes():
    assert _async_backed().assert_result().passed


def test_assert_result_raises_when_it_fails():
    with pytest.raises(AssertionError, match="does not exist within the response"):
        _async_backed(reference="Yours sincerely").assert_result()


def test_sync_api_works_inside_a_running_event_loop():
    """`asyncio.run` raises here, which is why the runner falls back to a thread.

    Consumers embed these in async services, not only in tests.
    """

    async def inside_a_loop():
        return _async_backed().evaluate().passed

    assert asyncio.run(inside_a_loop()) is True


def test_async_entry_points_are_public():
    """For callers already in a loop who would rather await than block it."""

    async def awaited():
        evaluator = _async_backed()
        result = await evaluator.evaluate_async()
        asserted = await evaluator.assert_result_async()
        return result.passed, asserted.passed

    assert asyncio.run(awaited()) == (True, True)


def test_failure_reason_reaches_the_assertion_message():
    """The judge and similarity evaluators explain themselves; that should surface."""
    result = EvalResult(name="demo", passed=False, reason="score 1.0 below 3.0")

    class Failing(RunJsonResponseEvaluator):
        def _evaluate(self):
            return result

    with pytest.raises(AssertionError, match="score 1.0 below 3.0"):
        Failing(response="{}").assert_result()
