"""The interface every evaluator shares.

Some evaluators await a scorer internally (ragas, azure-ai-evaluation) and some do
not. Both present the same synchronous API, so a consumer never has to look up which
is which — getting that wrong used to leave a coroutine un-awaited, so the assertion
never ran.

Subclass `Evaluator` and implement `_evaluate`, or `AsyncEvaluator` and implement
`_evaluate_async`.
"""

import asyncio
import atexit
import logging
import threading
from abc import ABC, abstractmethod
from typing import Any, Coroutine, Optional, TypeVar

from llm_eval.results import EvalResult
from llm_eval.tools.utils import format_dict_log

logger = logging.getLogger(__name__)

T = TypeVar("T")


_background_loop: Optional[asyncio.AbstractEventLoop] = None
_background_loop_lock = threading.Lock()


def _get_background_loop() -> asyncio.AbstractEventLoop:
    """The loop `run_sync` hands work to, started once and then reused.

    A thread and a loop per call cost more than the short scorers themselves, and
    this is the path every call from async code takes.
    """
    global _background_loop

    with _background_loop_lock:
        if _background_loop is None or _background_loop.is_closed():
            loop = asyncio.new_event_loop()
            threading.Thread(
                target=loop.run_forever,
                name="llm-eval-run-sync",
                daemon=True,
            ).start()
            _background_loop = loop
        return _background_loop


@atexit.register
def _stop_background_loop() -> None:
    """Let the loop finish rather than leaving the thread mid-callback at exit.

    The reference goes too: a stopped loop is not a closed one, so a later call
    would otherwise queue work on a loop nothing is running.
    """
    global _background_loop

    with _background_loop_lock:
        loop, _background_loop = _background_loop, None

    if loop is not None and not loop.is_closed():
        loop.call_soon_threadsafe(loop.stop)


def run_sync(coroutine: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine to completion from synchronous code.

    Safe inside a running event loop, where `asyncio.run` raises: the coroutine is
    scheduled on a shared background loop instead, and this thread blocks on its
    result. Consumers embed these evaluators in async services as well as tests.

    Args:
        coroutine (Coroutine): The coroutine to run.

    Returns:
        The coroutine's result.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    return asyncio.run_coroutine_threadsafe(coroutine, _get_background_loop()).result()


class Evaluator(ABC):
    """Base for evaluators whose scoring is synchronous.

    Attributes:
        name (str): Names the metric in the result and prefixes its keys in
            `EvalResult.raw`. Set as a class attribute where fixed, or in `__init__`
            where derived from the metric being wrapped.
        assertion_fail_message (str): Raised by `assert_result` on failure.
    """

    name: str = ""
    assertion_fail_message: str = "Evaluation failed"

    @abstractmethod
    def _evaluate(self) -> EvalResult:
        """Score the inputs. Implemented by each evaluator family."""

    async def _evaluate_async(self) -> EvalResult:
        """`_evaluate` in a thread, so awaiting it does not block the caller's loop.

        Synchronous scoring is still blocking work: a local model, or a network call.
        """
        return await asyncio.to_thread(self._evaluate)

    def evaluate(self) -> EvalResult:
        """Score the inputs and log the result.

        Returns:
            EvalResult: The outcome.
        """
        return self._log(self._evaluate())

    async def evaluate_async(self) -> EvalResult:
        """`evaluate`, for callers already inside an event loop."""
        return self._log(await self._evaluate_async())

    def assert_result(self) -> EvalResult:
        """Score the inputs, raising if they fail.

        Returns:
            EvalResult: The outcome, when it passed.

        Raises:
            AssertionError: If the evaluation failed.
        """
        return self._assert(self.evaluate())

    async def assert_result_async(self) -> EvalResult:
        """`assert_result`, for callers already inside an event loop."""
        return self._assert(await self.evaluate_async())

    def __call__(self) -> EvalResult:
        return self.evaluate()

    def _log(self, result: EvalResult) -> EvalResult:
        logger.info(format_dict_log(dictionary=self._log_payload(result)))
        return result

    def _log_payload(self, result: EvalResult) -> dict:
        """What to write to the log. Override to shorten large values."""
        return result.raw

    def _assert(self, result: EvalResult) -> EvalResult:
        if not result.passed:
            raise AssertionError(
                f"{self.assertion_fail_message}\n{result.reason}".rstrip()
            )
        return result


class AsyncEvaluator(Evaluator):
    """Base for evaluators whose scoring awaits something."""

    @abstractmethod
    async def _evaluate_async(self) -> EvalResult:
        """Score the inputs. Implemented by each evaluator family."""

    def _evaluate(self) -> EvalResult:
        return run_sync(self._evaluate_async())
