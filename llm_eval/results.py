"""The result type every evaluator returns."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class EvalResult(BaseModel):
    """The outcome of one evaluation.

    `passed` means the same thing for every evaluator, so reading a result does not
    depend on knowing which one produced it. Each previously returned a flat dict
    keyed by metric name (`faithfulness_result`, `similarity_result`, ...), where
    reading the wrong key gave `None` — a comparison that quietly never failed.

    `raw` keeps that flat dict verbatim, so anything already reading
    `{metric}_result` keys still works.

    Attributes:
        name (str): The metric, e.g. `"faithfulness"`.
        passed (bool): Whether the evaluation met its criteria.
        score (Optional[float]): The score, where the evaluator produces one.
        threshold (Optional[float]): The score needed to pass. None for binary
            metrics, which round the score instead of comparing.
        reason (str): Why it failed, where the evaluator explains itself.
        inputs (dict): The data that was evaluated.
        raw (dict): The flat metric dict, in the `{metric}_result` convention.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    passed: bool
    score: Optional[float] = None
    threshold: Optional[float] = None
    reason: str = ""
    inputs: dict = Field(default_factory=dict)
    raw: dict = Field(default_factory=dict)

    @property
    def result(self) -> str:
        """`"pass"` or `"fail"`, matching the wording used in `raw`."""
        return "pass" if self.passed else "fail"
