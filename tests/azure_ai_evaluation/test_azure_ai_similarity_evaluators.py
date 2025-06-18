from typing import Optional
import pytest
from llm_eval.azure_ai_evaluation.azure_ai_similarity_evaluators import (
    RunSimilarityEvaluator,
    AzureOpenAIModelConfiguration, RunF1ScoreEvaluator, RunBleuScoreEvaluator, RunGleuScoreEvaluator,
    RunRougeScoreEvaluator, RunMeteorScoreEvaluator,
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
def test_run_similarity_evaluator_should_pass(
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


@pytest.mark.parametrize(
    "query, response, ground_truth, threshold, model_config",
    [
        (
                "Is Marie Curie is born in Paris?",
                "According to wikipedia, Marie Curie was not born in Paris but in Warsaw.",
                "Marie Curie was born in London.",
                4.0,
                None,
        )
    ],
)
def test_run_similarity_evaluator_should_fail(
        query: str,
        response: str,
        ground_truth: str,
        threshold: float,
        model_config: Optional[AzureOpenAIModelConfiguration],
):
    with pytest.raises(AssertionError):
        RunSimilarityEvaluator(
            query=query,
            response=response,
            ground_truth=ground_truth,
            threshold=threshold,
            model_config=model_config,
        ).evaluate()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response, ground_truth, threshold",
    [
        (
                "According to wikipedia, Marie Curie was not born in Paris but in Warsaw.",
                "Marie Curie was born in Warsaw.",
                0.5
        ),
    ],
)
async def test_f1_score_evaluator_should_pass(response, ground_truth, threshold):
    await RunF1ScoreEvaluator(response, ground_truth, threshold).evaluate()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response, ground_truth, threshold",
    [
        (
                "According to wikipedia, Marie Curie was not born in Paris but in Warsaw.",
                "Marie Curie was born in Warsaw.",
                0.9
        ),
    ],
)
async def test_f1_score_evaluator_should_fail(response, ground_truth, threshold):
    with pytest.raises(AssertionError):
        await RunF1ScoreEvaluator(response, ground_truth, threshold).evaluate()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response, ground_truth, threshold",
    [
        (
                "Marie Curie was birthed in Warsaw.",
                "Marie Curie was born in Warsaw.",
                0.3
        ),
    ],
)
async def test_bleu_score_evaluator_should_pass(response, ground_truth, threshold):
    await RunBleuScoreEvaluator(response, ground_truth, threshold).evaluate()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response, ground_truth, threshold",
    [
        (
                "Marie Curie was birthed in Warsaw.",
                "Marie Curie was born in Warsaw.",
                0.4
        ),
    ],
)
async def test_bleu_score_evaluator_should_fail(response, ground_truth, threshold):
    with pytest.raises(AssertionError):
        await RunBleuScoreEvaluator(response, ground_truth, threshold).evaluate()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response, ground_truth, threshold",
    [
        (
                "Marie Curie was birthed in Warsaw.",
                "Marie Curie was born in Warsaw.",
                0.3
        ),
    ],
)
async def test_gleu_score_evaluator_should_pass(response, ground_truth, threshold):
    await RunGleuScoreEvaluator(response, ground_truth, threshold).evaluate()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response, ground_truth, threshold",
    [
        (
                "Marie Curie was birthed in Warsaw.",
                "Marie Curie was born in Warsaw.",
                0.6
        ),
    ],
)
async def test_gleu_score_evaluator_should_fail(response, ground_truth, threshold):
    with pytest.raises(AssertionError):
        await RunGleuScoreEvaluator(response, ground_truth, threshold).evaluate()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response, ground_truth, threshold",
    [
        (
                "According to wikipedia, Marie Curie was not born in Paris but in Warsaw.",
                "Marie Curie was born in Warsaw.",
                0.5
        ),
    ],
)
async def test_rouge_score_evaluator_should_pass(response, ground_truth, threshold):
    await RunRougeScoreEvaluator(response, ground_truth, threshold).evaluate()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response, ground_truth, threshold",
    [
        (
                "According to wikipedia, Marie Curie was not born in Paris but in Warsaw.",
                "Marie Curie was born in Warsaw.",
                0.7
        ),
    ],
)
async def test_rouge_score_evaluator_should_fail(response, ground_truth, threshold):
    with pytest.raises(AssertionError):
        await RunRougeScoreEvaluator(response, ground_truth, threshold).evaluate()

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response, ground_truth, threshold",
    [
        (
                "According to wikipedia, Marie Curie was not born in Paris but in Warsaw.",
                "Marie Curie was born in Warsaw.",
                0.7
        ),
    ],
)
async def test_meteor_score_evaluator_should_pass(response, ground_truth, threshold):
    await RunMeteorScoreEvaluator(response, ground_truth, threshold).evaluate()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response, ground_truth, threshold",
    [
        (
                "According to wikipedia, Marie Curie was not born in Paris but in Warsaw.",
                "Marie Curie was born in Warsaw.",
                0.9
        ),
    ],
)
async def test_rouge_score_evaluator_should_fail(response, ground_truth, threshold):
    with pytest.raises(AssertionError):
        await RunMeteorScoreEvaluator(response, ground_truth, threshold).evaluate()
