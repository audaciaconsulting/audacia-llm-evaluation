from typing import Optional
import pytest
from llm_eval.evaluators.similarity import (
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
    ).evaluate(assert_result=True)


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
        ).evaluate(assert_result=True)


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
    await RunF1ScoreEvaluator(response, ground_truth, threshold).evaluate(assert_result=True)


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
        await RunF1ScoreEvaluator(response, ground_truth, threshold).evaluate(assert_result=True)


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
    await RunBleuScoreEvaluator(response, ground_truth, threshold).evaluate(assert_result=True)


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
        await RunBleuScoreEvaluator(response, ground_truth, threshold).evaluate(assert_result=True)


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
    await RunGleuScoreEvaluator(response, ground_truth, threshold).evaluate(assert_result=True)


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
        await RunGleuScoreEvaluator(response, ground_truth, threshold).evaluate(assert_result=True)


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
    await RunRougeScoreEvaluator(response, ground_truth, threshold).evaluate(assert_result=True)


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
        await RunRougeScoreEvaluator(response, ground_truth, threshold).evaluate(assert_result=True)

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
    await RunMeteorScoreEvaluator(response, ground_truth, threshold).evaluate(assert_result=True)

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
        await RunMeteorScoreEvaluator(response, ground_truth, threshold).evaluate(assert_result=True)

import pytest

from llm_eval.evaluators.similarity import (
    RunExactMatch,
    RunNonLLMStringSimilarity,
    RunSemanticSimilarity,
    RunStringPresence,
)


@pytest.mark.asyncio
async def test_run_non_llm_string_similarity_passes(
    response="Marie Curie was birthed in Warsaw.",
    reference="Marie Curie was born in Warsaw.",
    threshold=0.8,
):
    await RunNonLLMStringSimilarity(response, reference, threshold).evaluate(assert_result=True)


@pytest.mark.asyncio
async def test_run_non_llm_string_similarity_fails(
    response="Einstein's groundbreaking theory of relativity transformed our comprehension of the cosmos",
    reference="Albert Einstein's theory of relativity revolutionized our understanding of the universe.",
    threshold=0.5,
):
    with pytest.raises(AssertionError):
        await RunNonLLMStringSimilarity(
            response, reference, threshold
        ).evaluate(assert_result=True)


@pytest.mark.asyncio
async def test_run_embedding_similarity_passes(
    response="Einstein's groundbreaking theory of relativity transformed our comprehension of the cosmos",
    reference="Albert Einstein's theory of relativity revolutionized our understanding of the universe.",
    threshold=0.8,
):
    await RunSemanticSimilarity(response, reference, threshold).evaluate(assert_result=True)


@pytest.mark.asyncio
async def test_run_embedding_similarity_fails(
    response="Isaac Newton's laws of motion greatly influenced classical physics",
    reference="Albert Einstein's theory of relativity revolutionized our understanding of the universe.",
    threshold=0.5,
):
    with pytest.raises(AssertionError):
        await RunSemanticSimilarity(response, reference, threshold).evaluate(assert_result=True)


@pytest.mark.asyncio
async def test_run_string_presence_passes(
    response="Einstein's groundbreaking theory of relativity transformed our comprehension of the cosmos",
    reference="relativity",
):
    await RunStringPresence(response, reference).evaluate(assert_result=True)


@pytest.mark.asyncio
async def test_run_string_presence_fails(
    response="Einstein's groundbreaking theory of relativity transformed our comprehension of the cosmos",
    reference="Newton",
):
    with pytest.raises(AssertionError):
        await RunStringPresence(response, reference).evaluate(assert_result=True)


@pytest.mark.asyncio
async def test_run_exact_match_passes(
    response="Marie Curie was born in Warsaw.",
    reference="Marie Curie was born in Warsaw.",
):
    await RunExactMatch(response, reference).evaluate(assert_result=True)


@pytest.mark.asyncio
async def test_run_exact_match_fails(
    response="Marie Curie was born in Paris.",
    reference="Marie Curie was born in Warsaw.",
):
    with pytest.raises(AssertionError):
        await RunExactMatch(response, reference).evaluate(assert_result=True)
