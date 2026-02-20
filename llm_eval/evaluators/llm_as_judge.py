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
    clean_prompt = str(prompt).replace('\n', ' - ')
    clean_prompt = re.sub(r'\s+', ' ', clean_prompt).strip()
    prompt_trunc = clean_prompt[:100]
    return prompt_trunc


def format_inputs(inputs: dict):
    return "## Inputs\n" + "\n\n".join(
        f"### {k}:\n{v}"
        for k, v in inputs.items()
    )


class JudgePassFailResult(BaseModel):
    llm_as_judge_result: Literal["pass", "fail"]
    failures_list: List[str] = Field(default_factory=list)


class RunLlmAsJudgePassFailEvaluator:
    """
    Evaluates a judge prompt using an LLM and returns a structured pass/fail result.

    The evaluator invokes a chat model with a strict output schema containing:
    - `llm_as_judge_result`: "pass" or "fail"
    - `failures_list`: list of failure reasons

    The supplied prompt must clearly instruct the model to produce an output that
    matches the `JudgePassFailResult` schema.
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
        inputs_str = format_inputs(self.inputs)
        structured_model = self.model.with_structured_output(JudgePassFailResult)
        llm_result = structured_model.invoke(
            [
                SystemMessage(content=self.prompt),
                HumanMessage(content=inputs_str),
            ]
        )
        result = {
            "llm_as_judge_result": llm_result.llm_as_judge_result,
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
    llm_as_judge_score: float
    failures_list: List[str] = Field(default_factory=list)


class RunLlmAsJudgeScoreEvaluator:
    """
    Evaluates a judge prompt using an LLM and returns a structured score result.

    The evaluator invokes a chat model with a strict output schema containing:
    - `llm_as_judge_score`: numeric score from the judge model
    - `failures_list`: list of failure reasons

    The supplied prompt must clearly instruct the model to produce an output that
    matches the `JudgeScoreResult` schema.
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
