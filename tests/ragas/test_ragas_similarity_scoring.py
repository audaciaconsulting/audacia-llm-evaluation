# from typing import Optional
# import pytest

# from ...llm_eval.ragas.ragas_similarity_evaluators import (
#     bleu_score,
#     embedding_similarity,
#     NonLLMStringSimilarityEval,
#     string_presence,
# )
# from llm_eval.models import get_azure_openai_llm_inference


# @pytest.mark.asyncio
# @pytest.mark.parametrize(
#     "prompt, reference, threshold, response",
#     [
#         (
#             "respond 'Hello World' only",
#             "Hello World",
#             1,
#             None,
#         ),
#         (
#             None,
#             "Testing non llm string similarity using a manual response",
#             0.9,
#             "Testing non llm string similarity with a manual response",
#         ),
#         (
#             None,
#             "Testing non llm string similarity using a manual response",
#             0.7,
#             "Evaluating non-LLM string similarity with manual responses",
#         ),
#     ],
# )
# async def test_response_for_non_llm_string_similarity(
#     prompt: str,
#     reference: str,
#     threshold: float,
#     response: Optional[str],
# ):
#     response = get_openai_llm_inference(prompt=prompt) if response is None else response
#     NonLLMStringSimilarityEval(
#         response=response, reference=reference, threshold=threshold
#     )


# @pytest.mark.timeout(30)
# @pytest.mark.asyncio
# @pytest.mark.parametrize(
#     "prompt, reference, threshold, response",
#     [
#         (
#             "hello",
#             "Hello! How can I assist you today?",
#             0.80,
#             None,
#         ),
#         (
#             None,
#             "She quickly closed the window before the storm came.",
#             0.75,
#             "She quickly closed the window before the storm arrived.",
#         ),
#         (
#             None,
#             "She quickly closed the window before the storm came.",
#             0.4,
#             "She quickly shut the window before the storm arrived.",
#         ),
#     ],
# )
# async def test_response_for_bleu_score(
#     prompt: str,
#     reference: str,
#     threshold: float,
#     response: Optional[str],
# ):
#     response = get_openai_llm_inference(prompt=prompt) if response is None else response
#     await bleu_score(response=response, reference=reference, threshold=threshold)


# # @pytest.mark.asyncio
# # @pytest.mark.parametrize(
# #     "prompt, reference, x_tool_selection, response",
# #     [
# #         (
# #             "hello",
# #             "assist you",
# #             SubApp.GUIDELINES_RISK,
# #             None,
# #         )
# #     ],
# # )
# # async def test_response_for_string_presence(
# #     prompt: str,
# #     reference: str,
# #     x_tool_selection: SubApp,
# #     response: Optional[str],
# #     endpoint: str = "/api/chat",
# # ):
# #     response = get_response_or_use_parameter(
# #         prompt=prompt,
# #         x_tool_selection=x_tool_selection,
# #         endpoint=endpoint,
# #         response=response,
# #     )
# #     await string_presence(response=response, reference=reference)


# # @pytest.mark.timeout(30)
# # @pytest.mark.asyncio
# # @pytest.mark.parametrize(
# #     "prompt, reference, x_tool_selection, threshold, response",
# #     [
# #         (
# #             "hello",
# #             "Good day! How may I support you?",
# #             SubApp.GUIDELINES_RISK,
# #             0.60,
# #             None,
# #         )
# #     ],
# # )
# # async def test_response_for_embedding_similarity(
# #     prompt: str,
# #     reference: str,
# #     x_tool_selection: SubApp,
# #     threshold: float,
# #     response: Optional[str],
# #     endpoint: str = "/api/chat"
# # ):
# #     response = get_response_or_use_parameter(
# #         prompt=prompt,
# #         x_tool_selection=x_tool_selection,
# #         endpoint=endpoint,
# #         response=response,
# #     )

# #     await embedding_similarity(
# #         response=response,
# #         reference=reference,
# #         threshold=threshold,
# #         embedding_model=get_embedding_model(),
# #     )
