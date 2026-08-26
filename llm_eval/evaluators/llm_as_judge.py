import re
from typing import Optional
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from llm_eval.base_evaluators.evaluator import Evaluator
from llm_eval.results import EvalResult
from llm_eval.tools.model_tools import get_azure_openai_llm

from pydantic import BaseModel, Field
from typing import List, Literal, Dict


def truncate_prompt(prompt):
    """Normalize and truncate prompt text for compact logging."""
    clean_prompt = str(prompt).replace('\n', ' - ')
    clean_prompt = re.sub(r'\s+', ' ', clean_prompt).strip()
    prompt_trunc = clean_prompt[:100]
    return prompt_trunc


def format_inputs(inputs: dict):
    """Render evaluator inputs into a markdown-like block for the judge."""
    return "## Inputs\n" + "\n\n".join(
        f"### {k}:\n{v}"
        for k, v in inputs.items()
    )


def check_prompt_output(prompt: str, inputs: dict):
    """
    Validate that the prompt keeps required template markers.

    Required prompt content:
    - `## Inputs`
    - `You will receive:`
    - numbered input labels matching `inputs` keys exactly, e.g. `1. query`, `2. response`
    - `## Output Format`
    - `Output ONLY valid JSON`
    - `llm_as_judge_score`
    - `failures_list`

    Raises:
        ValueError: If any required marker is missing.
    """
    inputs_strs = ["## Inputs", "You will receive:"]
    for i, (k, v) in enumerate(inputs.items(), start=1):
        inputs_strs.append(f'{i}. {k}')

    output_strs = [
        "## Output Format",
        "Output ONLY valid JSON",
        "llm_as_judge_score",
        "failures_list",
    ]

    required = inputs_strs + output_strs

    for s in required:
        if s not in prompt:
            raise ValueError(f"Expected {s!r} in prompt")


#: The scale the shipped score template defines; a custom template must keep it.
JUDGE_SCORE_MIN, JUDGE_SCORE_MAX = 0.0, 1.0


class JudgePassFailResult(BaseModel):
    """Structured output schema for pass/fail judge responses using a score field."""
    llm_as_judge_score: Literal["pass", "fail"]
    failures_list: List[str] = Field(default_factory=list)


class JudgeScoreResult(BaseModel):
    """Structured output schema for score-based judge responses."""
    llm_as_judge_score: float = Field(ge=JUDGE_SCORE_MIN, le=JUDGE_SCORE_MAX)
    failures_list: List[str] = Field(default_factory=list)


class BaseLlmAsJudgeEvaluator(Evaluator):
    """
    Evaluate prompt-defined criteria with an LLM judge.

    Subclasses set `result_schema` (the structured output the judge must return) and
    implement `_passed`, which decides on that output.

    Prompt template constraints enforced by `check_prompt_output`:
    - Keep these exact strings in the prompt:
      - `## Inputs`
      - `You will receive:`
      - `## Output Format`
      - `Output ONLY valid JSON`
      - `llm_as_judge_score`
      - `failures_list`
    - Keep numbered input lines whose labels exactly match the keys of `inputs`
      in order (e.g. `1. query`, `2. response`).
    - If you customize the template, do not remove or rename these required strings.

    The evaluator sends:
    - `prompt` as a system message containing judging instructions.
    - `inputs` as a formatted human message payload.

    `EvalResult.raw` includes:
    - `llm_as_judge_result` (`"pass"` or `"fail"`)
    - `failures_list`
    - `prompt_trunc`
    """

    name = "llm_as_judge"
    result_schema: type
    threshold: Optional[float] = None

    def __init__(
            self,
            prompt: str,
            inputs: Dict[str, str],
            model: Optional[AzureChatOpenAI] = None,
    ):
        """Initialize the judge evaluator.

        Args:
            prompt (str): Judging instructions, sent as the system message.
            inputs (Dict[str, str]): Inputs for the judge to consider, sent as the
                human message. Keys must match the prompt's numbered input labels.
            model (Optional[AzureChatOpenAI]): Judge model. Defaults to the
                environment-configured Azure OpenAI client.
        """
        self.prompt = prompt
        self.inputs = inputs
        self.model = model or get_azure_openai_llm()

    def _passed(self, judged) -> bool:
        """Whether the judge's output counts as a pass.

        Args:
            judged: The judge's response, parsed into `result_schema`.
        """
        raise NotImplementedError

    def _score(self, judged) -> Optional[float]:
        """The numeric score, where the judge reports one."""
        return None

    def _raw(self, judged, passed: bool) -> dict:
        """Extra `raw` fields specific to this judge."""
        return {}

    def _evaluate(self) -> EvalResult:
        check_prompt_output(self.prompt, self.inputs)
        structured_model = self.model.with_structured_output(self.result_schema)
        judged = structured_model.invoke(
            [
                SystemMessage(content=self.prompt),
                HumanMessage(content=format_inputs(self.inputs)),
            ]
        )

        passed = self._passed(judged)

        return EvalResult(
            name=self.name,
            passed=passed,
            score=self._score(judged),
            threshold=self.threshold,
            reason="; ".join(judged.failures_list),
            inputs=dict(self.inputs),
            raw={
                **self._raw(judged, passed),
                "llm_as_judge_result": "pass" if passed else "fail",
                "failures_list": judged.failures_list,
                "prompt_trunc": truncate_prompt(self.prompt),
            },
        )


class RunLlmAsJudgePassFailEvaluator(BaseLlmAsJudgeEvaluator):
    """
    Evaluate prompt-defined criteria with an LLM judge and return pass/fail output.

    Use the pass/fail template at:
    - `llm_eval/prompt_templates/llm-as-judge-template.md`

    See `BaseLlmAsJudgeEvaluator` for the prompt template constraints.

    Expected structured model output (`JudgePassFailResult`):
    - `llm_as_judge_score`: `"pass"` or `"fail"`
    - `failures_list`: list of failure reasons
    """

    assertion_fail_message = "LLM-as-Judge evaluation failed"
    result_schema = JudgePassFailResult

    def _passed(self, judged) -> bool:
        return judged.llm_as_judge_score == "pass"


class RunLlmAsJudgeScoreEvaluator(BaseLlmAsJudgeEvaluator):
    """
    Evaluate prompt-defined criteria with an LLM judge and return a scored result.

    Use the score-threshold template at:
    - `llm_eval/prompt_templates/llm-as-judge-score-threshold-template.md`

    See `BaseLlmAsJudgeEvaluator` for the prompt template constraints.

    Expected structured model output (`JudgeScoreResult`):
    - `llm_as_judge_score`: numeric score from the judge model
    - `failures_list`: list of failure reasons

    `EvalResult.raw` also includes `llm_as_judge_score` and `threshold`.
    """

    assertion_fail_message = "LLM-as-Judge score evaluation failed"
    result_schema = JudgeScoreResult

    def __init__(
            self,
            prompt: str,
            inputs: Dict[str, str],
            threshold: float,
            model: Optional[AzureChatOpenAI] = None,
    ):
        """Initialize the scoring judge evaluator.

        Args:
            prompt (str): Judging instructions, sent as the system message.
            inputs (Dict[str, str]): Inputs for the judge to consider, sent as the
                human message. Keys must match the prompt's numbered input labels.
            threshold (float): Minimum score the judge must give to pass, on the
                0.0-1.0 scale the score template defines.

        Raises:
            ValueError: If `threshold` falls outside that scale.
            model (Optional[AzureChatOpenAI]): Judge model. Defaults to the
                environment-configured Azure OpenAI client.
        """
        if not JUDGE_SCORE_MIN <= threshold <= JUDGE_SCORE_MAX:
            raise ValueError(
                f"Threshold must be between {JUDGE_SCORE_MIN} and {JUDGE_SCORE_MAX}, the "
                f"scale the score template defines. Got {threshold}."
            )

        self.threshold = threshold
        super().__init__(prompt=prompt, inputs=inputs, model=model)

    def _passed(self, judged) -> bool:
        return judged.llm_as_judge_score >= self.threshold

    def _score(self, judged) -> Optional[float]:
        return judged.llm_as_judge_score

    def _raw(self, judged, passed: bool) -> dict:
        return {
            "llm_as_judge_score": judged.llm_as_judge_score,
            "threshold": self.threshold,
        }
