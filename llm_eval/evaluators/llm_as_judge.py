import logging
import re
from typing import Optional
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from llm_eval.tools.model_tools import get_azure_openai_llm
from llm_eval.tools.utils import format_dict_log

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


class JudgePassFailResult(BaseModel):
    """Structured output schema for pass/fail judge responses using a score field."""
    llm_as_judge_score: Literal["pass", "fail"]
    failures_list: List[str] = Field(default_factory=list)


class RunLlmAsJudgePassFailEvaluator:
    """
    Evaluate prompt-defined criteria with an LLM judge and return pass/fail output.

    Use the pass/fail template at:
    - `llm_eval/prompt_templates/llm-as-judge-template.md`

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

    Expected structured model output (`JudgePassFailResult`):
    - `llm_as_judge_score`: `"pass"` or `"fail"`
    - `failures_list`: list of failure reasons

    Returned dict includes:
    - `llm_as_judge_result`
    - `failures_list`
    - `prompt_trunc`
    """

    def __init__(
            self,
            prompt: str,
            inputs: Dict[str, str],
            model: Optional[AzureChatOpenAI] = None,
    ):
        self.prompt = prompt
        self.model = model or get_azure_openai_llm()
        self.inputs = inputs

    def __call__(self) -> dict:
        check_prompt_output(self.prompt, self.inputs)
        inputs_str = format_inputs(self.inputs)
        structured_model = self.model.with_structured_output(JudgePassFailResult)
        llm_result = structured_model.invoke(
            [
                SystemMessage(content=self.prompt),
                HumanMessage(content=inputs_str),
            ]
        )
        result = {
            "llm_as_judge_result": llm_result.llm_as_judge_score,
            "failures_list": llm_result.failures_list,
            "prompt_trunc": truncate_prompt(self.prompt)
        }

        logger.info(format_dict_log(dictionary=result))
        return result

    def assert_result(self):
        result = self()
        if result.get("llm_as_judge_result") == "fail":
            raise AssertionError("LLM-as-Judge evaluation failed")

    def evaluate(self, assert_result: bool = False):
        result = self()

        logger.info(format_dict_log(dictionary=result))

        if assert_result:
            assert result["llm_as_judge_result"] == "pass"

        return result


class JudgeScoreResult(BaseModel):
    """Structured output schema for score-based judge responses."""
    llm_as_judge_score: float
    failures_list: List[str] = Field(default_factory=list)


class RunLlmAsJudgeScoreEvaluator:
    """
    Evaluate prompt-defined criteria with an LLM judge and return a scored result.

     Use the score-threshold template at:
    - `llm_eval/prompt_templates/llm-as-judge-score-threshold-template.md`

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

    Expected structured model output (`JudgeScoreResult`):
    - `llm_as_judge_score`: numeric score from the judge model
    - `failures_list`: list of failure reasons

    Returned dict includes:
    - `llm_as_judge_score`
    - `llm_as_judge_result` (`"pass"` when score `>= threshold`, else `"fail"`)
    - `threshold`
    - `failures_list`
    - `prompt_trunc`
    """

    def __init__(
            self,
            prompt: str,
            inputs: Dict[str, str],
            threshold: float,
            model: Optional[AzureChatOpenAI] = None,
    ):
        self.prompt = prompt
        self.inputs = inputs
        self.threshold = threshold
        self.model = model or get_azure_openai_llm()

    def __call__(self) -> dict:
        check_prompt_output(self.prompt, self.inputs)
        inputs_str = format_inputs(self.inputs)
        structured_model = self.model.with_structured_output(JudgeScoreResult)
        llm_result = structured_model.invoke(
            [
                SystemMessage(content=self.prompt),
                HumanMessage(content=inputs_str),
            ]
        )
        result = {
            "llm_as_judge_score": llm_result.llm_as_judge_score,
            "llm_as_judge_result": "pass" if llm_result.llm_as_judge_score >= self.threshold else "fail",
            "threshold": self.threshold,
            "failures_list": llm_result.failures_list,
            "prompt_trunc": truncate_prompt(self.prompt)
        }

        logger.info(format_dict_log(dictionary=result))
        return result

    def assert_result(self):
        result = self()
        if result.get("llm_as_judge_result") == "fail":
            raise AssertionError("LLM-as-Judge score evaluation failed")

    def evaluate(self, assert_result: bool = False):
        result = self()

        logger.info(format_dict_log(dictionary=result))

        if assert_result:
            assert result["llm_as_judge_result"] == "pass"

        return result
