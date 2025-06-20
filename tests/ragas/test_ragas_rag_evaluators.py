import pytest
from ragas_evaluators.ragas_rag_evaluators import RunLLMContextPrecisionWithReference, \
    RunNonLLMContextPrecisionWithReference, RunLLMContextRecall, RunNonLLMContextRecall


# Context Precision

@pytest.mark.asyncio
async def test_llm_context_precision_with_reference_passes(
        user_input="Where is the Eiffel Tower located?",
        reference="The Eiffel Tower is located in Paris.",
        retrieved_contexts=["Paris is the location of the Eiffel Tower"],
        threshold=0.9,

):
    await RunLLMContextPrecisionWithReference(user_input, reference, retrieved_contexts, threshold).evaluate_assert()


@pytest.mark.asyncio
async def test_llm_context_precision_with_reference_fails(
        user_input="Where is the Eiffel Tower located?",
        reference="The Eiffel Tower is located in Paris.",
        retrieved_contexts=["Big ben is located in London.", "Paris is the location of the Eiffel Tower", ],
        threshold=0.6,
):
    with pytest.raises(AssertionError):
        await RunLLMContextPrecisionWithReference(user_input, reference, retrieved_contexts,
                                                  threshold).evaluate_assert()


@pytest.mark.asyncio
async def test_non_llm_context_precision_with_reference_passes(
        retrieved_contexts=["The Eiffel Tower is located in Paris."],
        reference_contexts=["Paris is the capital of France.",
                            "The Eiffel Tower is one of the most famous landmarks in Paris."],
        threshold=0.8,

):
    await RunNonLLMContextPrecisionWithReference(retrieved_contexts, reference_contexts, threshold).evaluate_assert()


@pytest.mark.asyncio
async def test_non_llm_context_precision_with_reference_fails(
        retrieved_contexts=["Big Ben is located in London"],
        reference_contexts=["Paris is the capital of France.",
                            "The Eiffel Tower is one of the most famous landmarks in Paris."],
        threshold=0.8,

):
    with pytest.raises(AssertionError):
        await RunNonLLMContextPrecisionWithReference(retrieved_contexts, reference_contexts,
                                                     threshold).evaluate_assert()


# Context Recall

@pytest.mark.asyncio
async def test_llm_context_recall_passes(
        user_input="Where is the Eiffel Tower located?",
        response="The Eiffel Tower is located in Paris.",
        reference="The Eiffel Tower is located in Paris.",
        retrieved_contexts=["Paris is the capital of France and location of the Eiffel Tower."],
        threshold=0.8,

):
    await RunLLMContextRecall(user_input=user_input, response=response, reference=reference,
                              retrieved_contexts=retrieved_contexts, threshold=threshold).evaluate_assert()


@pytest.mark.asyncio
async def test_llm_context_recall_fails(
        user_input="Where is the Eiffel Tower located?",
        response="The Eiffel Tower is located in Paris.",
        reference="The Eiffel Tower is located in Paris.",
        retrieved_contexts=["London is the capital of England and location of Big Ben."],
        threshold=0.8,

):
    with pytest.raises(AssertionError):
        await RunLLMContextRecall(user_input=user_input, response=response, reference=reference,
                                  retrieved_contexts=retrieved_contexts, threshold=threshold).evaluate_assert()


@pytest.mark.asyncio
async def test_non_llm_context_recall_passes(
        retrieved_contexts=["Paris is the capital of France."],
        reference_contexts=["Paris is the capital of France.",
                            "The Eiffel Tower is one of the most famous landmarks in Paris."],
        threshold=0.5,

):
    await RunNonLLMContextRecall(retrieved_contexts=retrieved_contexts, reference_contexts=reference_contexts,
                                 threshold=threshold).evaluate_assert()


@pytest.mark.asyncio
async def test_non_llm_context_recall_fails(
        retrieved_contexts=["Paris is the capital of France."],
        reference_contexts=["Paris is the capital of France.",
                            "The Eiffel Tower is one of the most famous landmarks in Paris."],
        threshold=0.6,

):
    with pytest.raises(AssertionError):
        await RunNonLLMContextRecall(retrieved_contexts=retrieved_contexts, reference_contexts=reference_contexts,
                                     threshold=threshold).evaluate_assert()
