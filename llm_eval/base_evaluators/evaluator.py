"""The interface every evaluator shares.

Some evaluators await a scorer internally (ragas, azure-ai-evaluation) and some do
not. Both present the same synchronous API, so a consumer never has to look up which
is which — getting that wrong used to leave a coroutine un-awaited, so the assertion
never ran.

Subclass `Evaluator` and implement `_evaluate`, or `AsyncEvaluator` and implement
`_evaluate_async`.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Awaitable, TypeVar

from llm_eval.results import EvalResult
from llm_eval.tools.utils import format_dict_log

logger = logging.getLogger(__name__)

T = TypeVar("T")


def run_sync(awaitable: Awaitable[T]) -> T:
    """Run an awaitable to completion from synchronous code.

    Safe inside a running event loop, where `asyncio.run` raises: the coroutine goes
    to a worker thread with a loop of its own. Consumers embed these evaluators in
    async services as well as tests.

    Args:
        awaitable (Awaitable): The coroutine to run.

    Returns:
        The awaitable's result.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)  # type: ignore[arg-type]

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, awaitable).result()  # type: ignore[arg-type]


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
        return self._evaluate()

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
