# from ragas.dataset_schema import SingleTurnSample
# from ragas.metrics import NonLLMStringSimilarity, BleuScore, StringPresence
# from ragas.metrics._answer_similarity import AnswerSimilarity
# from ragas.embeddings import LangchainEmbeddingsWrapper
# import logging

# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)


# async def score_response(
#     response: str,
#     reference: str,
#     ragas_metric: type,
#     threshold: float | bool,
#     ragas_metric_args: dict = {},
# ):

#     sample = SingleTurnSample(
#         response=response,
#         reference=reference,
#     )

#     score = await ragas_metric(**ragas_metric_args).single_turn_ascore(sample=sample)

#     if isinstance(threshold, bool):
#         pass_eval = round(score) == 1
#     else:
#         pass_eval = score >= threshold

#     metric_name = ragas_metric.__name__
#     metric_name_threshold = f"{metric_name}_threshold"
#     metric_name_result = f"{metric_name}_result"

#     results_dict = {
#         "response": response,
#         "reference": reference,
#         metric_name: score,
#         metric_name_threshold: threshold,
#         metric_name_result: pass_eval,
#     }

#     # logger.info(
#     #     f"""\n
#     #       **************************
#     #       response: {response}
#     #       reference: {reference}
#     #       {metric_name}: {score}
#     #       {metric_name_threshold}: {threshold}
#     #       {metric_name_result}: {pass_eval}
#     #       **************************\n"""
#     # )

#     logger.info(
#         f"""\n
#           **************************
#           {results_dict}
#           **************************\n"""
#     )

#     return results_dict


# class StringPresenceEval:
#     # def __init__(self):
#     #     pass

#     async def __call__(self, response: str, reference: str):
#         self.response = response
#         self.reference = reference

#         return await score_response(
#             response=response,
#             reference=reference,
#             ragas_metric=StringPresence,
#             threshold=False,
#         )


# async def string_presence(response: str, reference: str):
#     await score_response(
#         response=response,
#         reference=reference,
#         ragas_metric=StringPresence,
#         threshold=False,
#     )


# class NonLLMStringSimilarityEval:

#     def __init__(self, response: str, reference: str, threshold: float):
#         self.response = response
#         self.reference = reference
#         self.threshold = threshold

#     async def __call__(self):
#         return await score_response(
#             response=self.response,
#             reference=self.reference,
#             ragas_metric=NonLLMStringSimilarity,
#             threshold=self.threshold,
#         )

#     async def evaluate(self):
#         result = self.__call__
#         print(result)


# async def bleu_score(
#     response: str,
#     reference: str,
#     threshold: float,
# ):
#     await score_response(
#         response=response,
#         reference=reference,
#         ragas_metric=BleuScore,
#         threshold=threshold,
#     )


# async def embedding_similarity(
#     response: str,
#     reference: str,
#     threshold: float,
#     embedding_model: LangchainEmbeddingsWrapper,
# ):
#     await score_response(
#         response=response,
#         reference=reference,
#         ragas_metric=AnswerSimilarity,
#         threshold=threshold,
#         ragas_metric_args={"embeddings": embedding_model},
#     )
