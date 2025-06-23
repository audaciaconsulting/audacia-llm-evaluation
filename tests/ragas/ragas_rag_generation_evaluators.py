import pytest

from ragas_evaluators.ragas_rag_generation_evaluators import RunFaithfulness, RunResponseRelevancy


@pytest.mark.asyncio
async def test_run_faithfulness_passes(
        user_input="When was the first super bowl?",
        response="The first superbowl was held on Jan 15, 1967",
        retrieved_contexts=[
            "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles."
        ],
        threshold=0.9):
    await RunFaithfulness(user_input=user_input, response=response, retrieved_contexts=retrieved_contexts,
                          threshold=threshold).evaluate_assert()


@pytest.mark.asyncio
async def test_run_faithfulness_fails(
        user_input="When was the first super bowl?",
        response="The first superbowl was held on Jan 15, 1967",
        retrieved_contexts=[
            "The First AFL–NFL European Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles."
        ],
        threshold=0.9):
    with pytest.raises(AssertionError):
        await RunFaithfulness(user_input=user_input, response=response, retrieved_contexts=retrieved_contexts,
                              threshold=threshold).evaluate_assert()


@pytest.mark.asyncio
async def test_run_response_relevancy_passes(
        user_input="When was the first super bowl?",
        response="The first superbowl was held on Jan 15, 1967",
        threshold=0.9):
    await RunResponseRelevancy(user_input=user_input, response=response,
                               threshold=threshold).evaluate_assert()


@pytest.mark.asyncio
async def test_run_response_relevancy_fails(
        user_input="When was the first super bowl?",
        response="The Super Bowl is a championship game in American football played annually.",
        threshold=0.7):
    with pytest.raises(AssertionError):
        await RunResponseRelevancy(user_input=user_input, response=response,
                                   threshold=threshold).evaluate_assert()
