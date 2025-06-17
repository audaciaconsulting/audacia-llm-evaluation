from typing import Optional
from azure.ai.evaluation import SimilarityEvaluator, AzureOpenAIModelConfiguration
from llm_eval.models import get_azure_ai_evaluation_model_config
import logging

from tests.utils import format_dict_log

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class RunSimilarityEvaluator:

    def __init__(
        self,
        query: str,
        response: str,
        ground_truth: str,
        threshold: float,
        model_config: Optional[AzureOpenAIModelConfiguration],
    ):
        self.query = query
        self.response = response
        self.ground_truth = ground_truth
        self.threshold = threshold
        self.model_config = model_config or get_azure_ai_evaluation_model_config()

    def __call__(self) -> dict:
        similarity = SimilarityEvaluator(model_config=self.model_config)
        return similarity(
            query=self.query,
            response=self.response,
            ground_truth=self.ground_truth,
        )

    def evaluate(self):

        result = self()
        result_threshold = result["gpt_similarity"] >= self.threshold

        result.update(
            {
                "query": self.query,
                "response": self.response,
                "ground_truth": self.ground_truth,
                "threshold": self.threshold,
                "similarity_result": "pass" if result_threshold else "fail",
            }
        )

        logger.info(format_dict_log(dictionary=result))
        assert result_threshold
