from typing import Optional

import pytest

from llm_eval.evaluators.similarity import (
    AzureOpenAIModelConfiguration,
    RunBleuScoreEvaluator,
    RunExactMatch,
    RunF1ScoreEvaluator,
    RunGleuScoreEvaluator,
    RunMeteorScoreEvaluator,
    RunNonLLMStringSimilarity,
    RunRougeScoreEvaluator,
    RunSemanticSimilarity,
    RunSimilarityEvaluator,
    RunStringPresence,
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
    evaluator = RunSimilarityEvaluator(
        query=query,
        response=response,
        ground_truth=ground_truth,
        threshold=threshold,
        model_config=model_config,
    )

    result = evaluator()
    evaluator.assert_result()
    assert all(
        key in result
        for key in [
            "similarity",
            "gpt_similarity",
            "similarity_result",
            "similarity_threshold",
        ]
    )


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
        evaluator = RunSimilarityEvaluator(
            query=query,
            response=response,
            ground_truth=ground_truth,
            threshold=threshold,
            model_config=model_config,
        )

        evaluator.assert_result()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response, ground_truth, threshold",
    [
        (
            "According to wikipedia, Marie Curie was not born in Paris but in Warsaw.",
            "Marie Curie was born in Warsaw.",
            0.5,
        ),
    ],
)
async def test_f1_score_evaluator_should_pass(response, ground_truth, threshold):
    evaluator = RunF1ScoreEvaluator(response, ground_truth, threshold)
    result = await evaluator()
    await evaluator.assert_result()
    assert all(key in result for key in ["f1_score", "f1_result", "f1_threshold"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response, ground_truth, threshold",
    [
        (
            "According to wikipedia, Marie Curie was not born in Paris but in Warsaw.",
            "Marie Curie was born in Warsaw.",
            0.9,
        ),
    ],
)
async def test_f1_score_evaluator_should_fail(response, ground_truth, threshold):
    evaluator = RunF1ScoreEvaluator(response, ground_truth, threshold)
    with pytest.raises(AssertionError):
        await evaluator.assert_result()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response, ground_truth, threshold",
    [
        ("Marie Curie was birthed in Warsaw.", "Marie Curie was born in Warsaw.", 0.3),
    ],
)
async def test_bleu_score_evaluator_should_pass(response, ground_truth, threshold):
    evaluator = RunBleuScoreEvaluator(response, ground_truth, threshold)
    result = await evaluator()
    await evaluator.assert_result()
    assert all(key in result for key in ["bleu_score", "bleu_result", "bleu_threshold"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response, ground_truth, threshold",
    [
        ("Marie Curie was birthed in Warsaw.", "Marie Curie was born in Warsaw.", 0.4),
    ],
)
async def test_bleu_score_evaluator_should_fail(response, ground_truth, threshold):
    with pytest.raises(AssertionError):
        await RunBleuScoreEvaluator(response, ground_truth, threshold).assert_result()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response, ground_truth, threshold",
    [
        ("Marie Curie was birthed in Warsaw.", "Marie Curie was born in Warsaw.", 0.3),
    ],
)
async def test_gleu_score_evaluator_should_pass(response, ground_truth, threshold):
    evaluator = RunGleuScoreEvaluator(response, ground_truth, threshold)
    result = await evaluator()
    await evaluator.assert_result()
    assert all(key in result for key in ["gleu_score", "gleu_result", "gleu_threshold"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response, ground_truth, threshold",
    [
        ("Marie Curie was birthed in Warsaw.", "Marie Curie was born in Warsaw.", 0.6),
    ],
)
async def test_gleu_score_evaluator_should_fail(response, ground_truth, threshold):
    with pytest.raises(AssertionError):
        await RunGleuScoreEvaluator(response, ground_truth, threshold).assert_result()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response, ground_truth, threshold",
    [
        (
            "According to wikipedia, Marie Curie was not born in Paris but in Warsaw.",
            "Marie Curie was born in Warsaw.",
            0.5,
        ),
    ],
)
async def test_rouge_score_evaluator_should_pass(response, ground_truth, threshold):
    evaluator = RunRougeScoreEvaluator(response, ground_truth, threshold)
    result = await evaluator()
    await evaluator.assert_result()
    assert all(
        key in result
        for key in [
            "rouge_precision",
            "rouge_recall",
            "rouge_f1_score",
            "rouge_precision_result",
            "rouge_recall_result",
            "rouge_f1_score_result",
            "rouge_precision_threshold",
            "rouge_recall_threshold",
            "rouge_f1_score_threshold",
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response, ground_truth, threshold",
    [
        (
            "According to wikipedia, Marie Curie was not born in Paris but in Warsaw.",
            "Marie Curie was born in Warsaw.",
            0.7,
        ),
    ],
)
async def test_rouge_score_evaluator_should_fail(response, ground_truth, threshold):
    with pytest.raises(AssertionError):
        await RunRougeScoreEvaluator(response, ground_truth, threshold).assert_result()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response, ground_truth, threshold",
    [
        (
            "According to wikipedia, Marie Curie was not born in Paris but in Warsaw.",
            "Marie Curie was born in Warsaw.",
            0.7,
        ),
    ],
)
async def test_meteor_score_evaluator_should_pass(response, ground_truth, threshold):
    evaluator = RunMeteorScoreEvaluator(response, ground_truth, threshold)
    result = await evaluator()
    await evaluator.assert_result()
    assert all(
        key in result for key in ["meteor_score", "meteor_result", "meteor_threshold"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response, ground_truth, threshold",
    [
        (
            "According to wikipedia, Marie Curie was not born in Paris but in Warsaw.",
            "Marie Curie was born in Warsaw.",
            0.9,
        ),
    ],
)
async def test_meteor_score_evaluator_should_fail(response, ground_truth, threshold):
    with pytest.raises(AssertionError):
        await RunMeteorScoreEvaluator(response, ground_truth, threshold).assert_result()


@pytest.mark.asyncio
async def test_run_non_llm_string_similarity_passes(
    response="Marie Curie was birthed in Warsaw.",
    reference="Marie Curie was born in Warsaw.",
    threshold=0.8,
):
    evaluator = RunNonLLMStringSimilarity(response, reference, threshold)
    result = await evaluator()
    await evaluator.assert_result()
    assert all(
        key in result
        for key in [
            "response",
            "reference",
            "non_llmstring_similarity",
            "non_llmstring_similarity_threshold",
            "non_llmstring_similarity_result",
        ]
    )


@pytest.mark.asyncio
async def test_run_non_llm_string_similarity_fails(
    response="Einstein's groundbreaking theory of relativity transformed our comprehension of the cosmos",
    reference="Albert Einstein's theory of relativity revolutionized our understanding of the universe.",
    threshold=0.5,
):
    with pytest.raises(AssertionError):
        await RunNonLLMStringSimilarity(response, reference, threshold).assert_result()


@pytest.mark.asyncio
async def test_run_embedding_similarity_passes(
    response="Einstein's groundbreaking theory of relativity transformed our comprehension of the cosmos",
    reference="Albert Einstein's theory of relativity revolutionized our understanding of the universe.",
    threshold=0.8,
):
    evaluator = RunSemanticSimilarity(response, reference, threshold)
    result = await evaluator()
    await evaluator.assert_result()
    assert all(
        key in result
        for key in [
            "response",
            "reference",
            "semantic_similarity",
            "semantic_similarity_threshold",
            "semantic_similarity_result",
        ]
    )


@pytest.mark.asyncio
async def test_run_embedding_similarity_fails(
    response="Isaac Newton's laws of motion greatly influenced classical physics",
    reference="Albert Einstein's theory of relativity revolutionized our understanding of the universe.",
    threshold=0.5,
):
    with pytest.raises(AssertionError):
        await RunSemanticSimilarity(response, reference, threshold).assert_result()


@pytest.mark.asyncio
async def test_run_string_presence_passes(
    response="Einstein's groundbreaking theory of relativity transformed our comprehension of the cosmos",
    reference="relativity",
):
    evaluator = RunStringPresence(response, reference)
    result = await evaluator()
    await evaluator.assert_result()
    assert all(
        key in result
        for key in [
            "response",
            "reference",
            "string_presence",
            "string_presence_threshold",
            "string_presence_result",
        ]
    )


@pytest.mark.asyncio
async def test_run_string_presence_fails(
    response="Einstein's groundbreaking theory of relativity transformed our comprehension of the cosmos",
    reference="Newton",
):
    with pytest.raises(AssertionError):
        await RunStringPresence(response, reference).assert_result()


@pytest.mark.asyncio
async def test_run_exact_match_passes(
    response="Marie Curie was born in Warsaw.",
    reference="Marie Curie was born in Warsaw.",
):
    evaluator = RunExactMatch(response, reference)
    result = await evaluator()
    await evaluator.assert_result()
    assert all(
        key in result
        for key in [
            "response",
            "reference",
            "exact_match",
            "exact_match_threshold",
            "exact_match_result",
        ]
    )


@pytest.mark.asyncio
async def test_run_exact_match_fails(
    response="Marie Curie was born in Paris.",
    reference="Marie Curie was born in Warsaw.",
):
    with pytest.raises(AssertionError):
        await RunExactMatch(response, reference).assert_result()
