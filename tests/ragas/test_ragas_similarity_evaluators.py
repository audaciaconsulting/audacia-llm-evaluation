import pytest

from llm_eval.ragas_evaluators.ragas_similarity_evaluators import RunNonLLMStringSimilarity, RunSemanticSimilarity, \
    RunStringPresence
from ragas_evaluators.ragas_similarity_evaluators import RunExactMatch


@pytest.mark.asyncio
async def test_run_non_llm_string_similarity_passes(
        response="Marie Curie was birthed in Warsaw.",
        reference="Marie Curie was born in Warsaw.",
        threshold=0.8,

):
    await RunNonLLMStringSimilarity(response, reference, threshold).evaluate_assert()

@pytest.mark.asyncio
async def test_run_non_llm_string_similarity_fails(
        response="Einstein's groundbreaking theory of relativity transformed our comprehension of the cosmos",
        reference="Albert Einstein's theory of relativity revolutionized our understanding of the universe.",
        threshold=0.5

):
    with pytest.raises(AssertionError):
        await RunNonLLMStringSimilarity(response, reference, threshold).evaluate_assert()


@pytest.mark.asyncio
async def test_run_embedding_similarity_passes(
        response="Einstein's groundbreaking theory of relativity transformed our comprehension of the cosmos",
        reference="Albert Einstein's theory of relativity revolutionized our understanding of the universe.",
        threshold=0.8
):
    await RunSemanticSimilarity(response, reference, threshold).evaluate_assert()


@pytest.mark.asyncio
async def test_run_embedding_similarity_fails(
        response="Isaac Newton's laws of motion greatly influenced classical physics",
        reference="Albert Einstein's theory of relativity revolutionized our understanding of the universe.",
        threshold=0.5
):
    with pytest.raises(AssertionError):
        await RunSemanticSimilarity(response, reference, threshold).evaluate_assert()

@pytest.mark.asyncio
async def test_run_string_presence_passes(
        response="Einstein's groundbreaking theory of relativity transformed our comprehension of the cosmos",
        reference="relativity",
):
    await RunStringPresence(response, reference).evaluate_assert()


@pytest.mark.asyncio
async def test_run_string_presence_fails(
        response="Einstein's groundbreaking theory of relativity transformed our comprehension of the cosmos",
        reference="Newton"
):
    with pytest.raises(AssertionError):
        await RunStringPresence(response, reference).evaluate_assert()

@pytest.mark.asyncio
async def test_run_exact_match_passes(
        response="Marie Curie was born in Warsaw.",
        reference="Marie Curie was born in Warsaw.",
):
    await RunExactMatch(response, reference).evaluate_assert()


@pytest.mark.asyncio
async def test_run_exact_match_fails(
        response="Marie Curie was born in Paris.",
        reference="Marie Curie was born in Warsaw.",
):
    with pytest.raises(AssertionError):
        await RunExactMatch(response, reference).evaluate_assert()
