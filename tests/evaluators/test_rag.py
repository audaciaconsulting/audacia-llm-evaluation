import pytest

from llm_eval.evaluators.rag import (
    RunLLMContextPrecisionWithReferenceEvaluator,
    RunLLMContextRecallEvaluator,
    RunNonLLMContextPrecisionWithReferenceEvaluator,
    RunNonLLMContextRecallEvaluator,
    RunFaithfulnessEvaluator,
    RunResponseRelevancyEvaluator,
)


@pytest.mark.asyncio
async def test_llm_context_precision_with_reference_passes(
    user_input="Where is the Eiffel Tower located?",
    reference="The Eiffel Tower is located in Paris.",
    retrieved_contexts=["Paris is the location of the Eiffel Tower"],
    threshold=0.9,
):
    await RunLLMContextPrecisionWithReferenceEvaluator(
        user_input, reference, retrieved_contexts, threshold
    ).assert_result()


@pytest.mark.asyncio
async def test_llm_context_precision_with_reference_output(
    user_input="Where is the Eiffel Tower located?",
    reference="The Eiffel Tower is located in Paris.",
    retrieved_contexts=["Paris is the location of the Eiffel Tower"],
    threshold=0.9,
):
    evaluator = RunLLMContextPrecisionWithReferenceEvaluator(
        user_input, reference, retrieved_contexts, threshold
    )
    result = await evaluator()

    assert all(
        key in result
        for key in [
            "user_input",
            "reference",
            "retrieved_contexts",
            "llmcontext_precision_with_reference",
            "llmcontext_precision_with_reference_threshold",
            "llmcontext_precision_with_reference_result",
        ]
    )


@pytest.mark.asyncio
async def test_llm_context_precision_with_reference_fails(
    user_input="Where is the Eiffel Tower located?",
    reference="The Eiffel Tower is located in Paris.",
    retrieved_contexts=[
        "Big ben is located in London.",
        "Paris is the location of the Eiffel Tower",
    ],
    threshold=0.6,
):
    with pytest.raises(AssertionError):
        await RunLLMContextPrecisionWithReferenceEvaluator(
            user_input, reference, retrieved_contexts, threshold
        ).assert_result()


@pytest.mark.asyncio
async def test_non_llm_context_precision_with_reference_passes(
    retrieved_contexts=["The Eiffel Tower is located in Paris."],
    reference_contexts=[
        "Paris is the capital of France.",
        "The Eiffel Tower is one of the most famous landmarks in Paris.",
    ],
    threshold=0.8,
):
    await RunNonLLMContextPrecisionWithReferenceEvaluator(
        retrieved_contexts, reference_contexts, threshold
    ).assert_result()


@pytest.mark.asyncio
async def test_non_llm_context_precision_with_reference_output(
    retrieved_contexts=["The Eiffel Tower is located in Paris."],
    reference_contexts=[
        "Paris is the capital of France.",
        "The Eiffel Tower is one of the most famous landmarks in Paris.",
    ],
    threshold=0.8,
):
    evaluator = RunNonLLMContextPrecisionWithReferenceEvaluator(
        retrieved_contexts, reference_contexts, threshold
    )
    result = await evaluator()

    assert all(
        key in result
        for key in [
            "retrieved_contexts",
            "reference_contexts",
            "non_llmcontext_precision_with_reference",
            "non_llmcontext_precision_with_reference_threshold",
            "non_llmcontext_precision_with_reference_result",
        ]
    )


@pytest.mark.asyncio
async def test_non_llm_context_precision_with_reference_fails(
    retrieved_contexts=["Big Ben is located in London"],
    reference_contexts=[
        "Paris is the capital of France.",
        "The Eiffel Tower is one of the most famous landmarks in Paris.",
    ],
    threshold=0.8,
):
    with pytest.raises(AssertionError):
        await RunNonLLMContextPrecisionWithReferenceEvaluator(
            retrieved_contexts, reference_contexts, threshold
        ).assert_result()


@pytest.mark.asyncio
async def test_llm_context_recall_passes(
    user_input="Where is the Eiffel Tower located?",
    response="The Eiffel Tower is located in Paris.",
    reference="The Eiffel Tower is located in Paris.",
    retrieved_contexts=[
        "Paris is the capital of France and location of the Eiffel Tower."
    ],
    threshold=0.8,
):
    await RunLLMContextRecallEvaluator(
        user_input=user_input,
        response=response,
        reference=reference,
        retrieved_contexts=retrieved_contexts,
        threshold=threshold,
    ).assert_result()


@pytest.mark.asyncio
async def test_llm_context_recall_output(
    user_input="Where is the Eiffel Tower located?",
    response="The Eiffel Tower is located in Paris.",
    reference="The Eiffel Tower is located in Paris.",
    retrieved_contexts=[
        "Paris is the capital of France and location of the Eiffel Tower."
    ],
    threshold=0.8,
):
    evaluator = RunLLMContextRecallEvaluator(
        user_input=user_input,
        response=response,
        reference=reference,
        retrieved_contexts=retrieved_contexts,
        threshold=threshold,
    )
    result = await evaluator()

    assert all(
        key in result
        for key in [
            "user_input",
            "response",
            "reference",
            "retrieved_contexts",
            "llmcontext_recall",
            "llmcontext_recall_threshold",
            "llmcontext_recall_result",
        ]
    )


@pytest.mark.asyncio
async def test_llm_context_recall_fails(
    user_input="Where is the Eiffel Tower located?",
    response="The Eiffel Tower is located in Paris.",
    reference="The Eiffel Tower is located in Paris.",
    retrieved_contexts=["London is the capital of England and location of Big Ben."],
    threshold=0.8,
):
    with pytest.raises(AssertionError):
        await RunLLMContextRecallEvaluator(
            user_input=user_input,
            response=response,
            reference=reference,
            retrieved_contexts=retrieved_contexts,
            threshold=threshold,
        ).assert_result()


@pytest.mark.asyncio
async def test_non_llm_context_recall_passes(
    retrieved_contexts=["Paris is the capital of France."],
    reference_contexts=[
        "Paris is the capital of France.",
        "The Eiffel Tower is one of the most famous landmarks in Paris.",
    ],
    threshold=0.5,
):
    await RunNonLLMContextRecallEvaluator(
        retrieved_contexts=retrieved_contexts,
        reference_contexts=reference_contexts,
        threshold=threshold,
    ).assert_result()


@pytest.mark.asyncio
async def test_non_llm_context_recall_output(
    retrieved_contexts=["Paris is the capital of France."],
    reference_contexts=[
        "Paris is the capital of France.",
        "The Eiffel Tower is one of the most famous landmarks in Paris.",
    ],
    threshold=0.5,
):
    evaluator = RunNonLLMContextRecallEvaluator(
        retrieved_contexts=retrieved_contexts,
        reference_contexts=reference_contexts,
        threshold=threshold,
    )
    result = await evaluator()

    assert all(
        key in result
        for key in [
            "retrieved_contexts",
            "reference_contexts",
            "non_llmcontext_recall",
            "non_llmcontext_recall_threshold",
            "non_llmcontext_recall_result",
        ]
    )


@pytest.mark.asyncio
async def test_non_llm_context_recall_fails(
    retrieved_contexts=["Paris is the capital of France."],
    reference_contexts=[
        "Paris is the capital of France.",
        "The Eiffel Tower is one of the most famous landmarks in Paris.",
    ],
    threshold=0.6,
):
    with pytest.raises(AssertionError):
        await RunNonLLMContextRecallEvaluator(
            retrieved_contexts=retrieved_contexts,
            reference_contexts=reference_contexts,
            threshold=threshold,
        ).assert_result()


@pytest.mark.asyncio
async def test_run_faithfulness_passes(
    user_input="When was the first super bowl?",
    response="The first superbowl was held on Jan 15, 1967",
    retrieved_contexts=[
        "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles."
    ],
    threshold=0.9,
):
    await RunFaithfulnessEvaluator(
        user_input=user_input,
        response=response,
        retrieved_contexts=retrieved_contexts,
        threshold=threshold,
    ).assert_result()


@pytest.mark.asyncio
async def test_run_faithfulness_output(
    user_input="When was the first super bowl?",
    response="The first superbowl was held on Jan 15, 1967",
    retrieved_contexts=[
        "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles."
    ],
    threshold=0.9,
):
    evaluator = RunFaithfulnessEvaluator(
        user_input=user_input,
        response=response,
        retrieved_contexts=retrieved_contexts,
        threshold=threshold,
    )
    result = await evaluator()

    assert all(
        key in result
        for key in [
            "user_input",
            "response",
            "retrieved_contexts",
            "faithfulness",
            "faithfulness_threshold",
            "faithfulness_result",
        ]
    )


@pytest.mark.asyncio
async def test_run_faithfulness_fails(
    user_input="When was the first super bowl?",
    response="The first superbowl was held on Jan 15, 1967",
    retrieved_contexts=[
        "The First AFL–NFL European Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles."
    ],
    threshold=0.9,
):
    with pytest.raises(AssertionError):
        await RunFaithfulnessEvaluator(
            user_input=user_input,
            response=response,
            retrieved_contexts=retrieved_contexts,
            threshold=threshold,
        ).assert_result()


@pytest.mark.asyncio
async def test_run_response_relevancy_passes(
    user_input="When was the first super bowl?",
    response="The first superbowl was held on Jan 15, 1967",
    threshold=0.9,
):
    await RunResponseRelevancyEvaluator(
        user_input=user_input, response=response, threshold=threshold
    ).assert_result()


@pytest.mark.asyncio
async def test_run_response_relevancy_output(
    user_input="When was the first super bowl?",
    response="The first superbowl was held on Jan 15, 1967",
    threshold=0.9,
):
    evaluator = RunResponseRelevancyEvaluator(
        user_input=user_input, response=response, threshold=threshold
    )
    result = await evaluator()

    assert all(
        key in result
        for key in [
            "user_input",
            "response",
            "response_relevancy",
            "response_relevancy_threshold",
            "response_relevancy_result",
        ]
    )


@pytest.mark.asyncio
async def test_run_response_relevancy_fails(
    user_input="When was the first super bowl?",
    response="The Super Bowl is a championship game in American football played annually.",
    threshold=0.7,
):
    with pytest.raises(AssertionError):
        await RunResponseRelevancyEvaluator(
            user_input=user_input, response=response, threshold=threshold
        ).assert_result()
