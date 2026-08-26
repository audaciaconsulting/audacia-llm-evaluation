"""Helper for asserting the flat metric dict an evaluator reports.

`EvalResult.raw` keeps the `{metric}` / `{metric}_result` / `{metric}_threshold`
convention that consumers and Azure tooling read. Each evaluator's test names the
keys it must report, so the contract stays documented without a ten-line block
crowding out the assertions the test is really about.
"""

from llm_eval.results import EvalResult


def assert_raw_keys(result: EvalResult, *keys: str) -> None:
    """Assert `result.raw` reports every one of `keys`.

    Args:
        result (EvalResult): The evaluation outcome.
        *keys (str): Keys the evaluator's flat dict must contain.

    Raises:
        AssertionError: Naming the keys that are missing.
    """
    missing = [key for key in keys if key not in result.raw]
    assert not missing, (
        f"{result.name} did not report {', '.join(missing)} in raw. "
        f"Got: {', '.join(sorted(result.raw))}."
    )
