import logging
from typing import Optional

from langchain_openai import AzureChatOpenAI

from llm_eval.tools.model_tools import get_azure_openai_llm
from llm_eval.tools.utils import format_dict_log

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field
from typing import List, Literal


class JudgePassFailResult(BaseModel):
    llm_as_judge_result: Literal["pass", "fail"]
    failures_list: List[str] = Field(default_factory=list)


class RunLlmAsJudgePassFailEvaluator:
    def __init__(
            self,
            prompt: str,
            model: Optional[AzureChatOpenAI] = None,
    ):
        self.prompt = prompt
        self.model = model or get_azure_openai_llm()

    def __call__(self) -> dict:
        structured_model = self.model.with_structured_output(JudgePassFailResult)
        llm_result = structured_model.invoke(self.prompt)
        result = {
            "llm_as_judge_result": llm_result.llm_as_judge_result,
            "failures_list": llm_result.failures_list,
            "prompt_trunc": str(self.prompt).replace('  ', '')[:100],
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
