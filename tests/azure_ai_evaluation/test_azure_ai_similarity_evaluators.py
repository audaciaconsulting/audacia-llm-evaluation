from typing import Optional
import pytest
from llm_eval.azure_ai_evaluation.azure_ai_similarity_evaluators import (
    RunSimilarityEvaluator,
    AzureOpenAIModelConfiguration,
)


@pytest.mark.parametrize(
    "query, response, ground_truth, threshold, model_config",
    [
        (
            "Is Marie Curie is born in Paris?",
            "According to wikipedia, Marie Curie was not born in Paris but in Warsaw.",
            "Marie Curie was born in Warsaw.",
            4.5,
            None,
        )
    ],
)
def test_run_similarity_evaluator_passes(
    query: str,
    response: str,
    ground_truth: str,
    threshold: float,
    model_config: Optional[AzureOpenAIModelConfiguration],
):
    RunSimilarityEvaluator(
        query=query,
        response=response,
        ground_truth=ground_truth,
        threshold=threshold,
        model_config=model_config,
    ).evaluate()
