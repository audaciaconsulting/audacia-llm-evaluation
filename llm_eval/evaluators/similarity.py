from typing import Optional
from azure.ai.evaluation import (
    SimilarityEvaluator,
    AzureOpenAIModelConfiguration,
    F1ScoreEvaluator,
    BleuScoreEvaluator,
    GleuScoreEvaluator,
    RougeScoreEvaluator,
    RougeType,
    MeteorScoreEvaluator,

)

from ragas.metrics import NonLLMStringSimilarity, StringPresence, ExactMatch, SemanticSimilarity
from ragas.embeddings import LangchainEmbeddingsWrapper
from llm_eval.tools.model_tools import get_ragas_wrapped_azure_open_ai_embedding_model
from llm_eval.base_evaluators.ragas_base_evaluator import RagasBaseEvaluator

from llm_eval.base_evaluators.azure_ai_similarity_base_evaluator import BaseScoreEvaluator
from llm_eval.tools.model_tools import get_azure_ai_evaluation_model_config
import logging

from llm_eval.tools.utils import format_dict_log

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RunSimilarityEvaluator:
    """
        Evaluation Class: Similarity
        Evaluation Method: Embedding/Cosine Similarity
        Granularity: Medium

        The similarity score is calculated using sentence-level embeddings to compare the semantic meaning
        between a model-generated response and a predefined ground truth. The score ranges from 1.0 (least similar) to 5.0 (most similar),
        indicating how closely the response matches the expected output in terms of meaning. The `query` provides context to ensure accurate comparison between `response` and `ground_truth`,
        as meaning can vary with the prompt.

        This evaluation method is useful for tasks that involve comparing AI-generated responses to reference answers,
        such as question answering, summarization, and other natural language generation tasks. It helps in validating
        and benchmarking model performance on semantic accuracy.

        Attributes:
            query (str): The input query or prompt providing context for comparison.
            response (str): The model-generated response to evaluate.
            ground_truth (str): The expected correct response or reference text.
            threshold (float): The minimum similarity score required to pass the evaluation, ranging from 0.0 to 5.0.
            model_config (Optional[AzureOpenAIModelConfiguration]): Configuration for the embedding model. If not provided,
                a default Azure AI configuration will be used.
        """

    def __init__(
            self,
            query: str,
            response: str,
            ground_truth: str,
            threshold: float,
            model_config: Optional[AzureOpenAIModelConfiguration],
    ):
        self.query = query
        self.response = response
        self.ground_truth = ground_truth
        self.threshold = threshold
        self.model_config = model_config or get_azure_ai_evaluation_model_config()

        if not 0.0 <= threshold <= 5.0:
            raise ValueError(f"Threshold must be between 0 and 5. Got {threshold}.")

    def __call__(self) -> dict:
        evaluator = SimilarityEvaluator(model_config=self.model_config, threshold=self.threshold)
        return evaluator(
            query=self.query,
            response=self.response,
            ground_truth=self.ground_truth,
        )

    def evaluate(self, assert_result: bool = False):
        result = self()

        result.update(
            {
                "query": self.query,
                "response": self.response,
                "ground_truth": self.ground_truth
            }
        )

        logger.info(format_dict_log(dictionary=result))

        if assert_result:
            assert result['similarity_result'] == 'pass'

        return result


class RunF1ScoreEvaluator(BaseScoreEvaluator):
    """
        Evaluation Class: Similarity
        Evaluation Method: Token
        Granularity: Low

        The F1 score is a harmonic mean of precision and recall. It measures the accuracy of a generated response
        by comparing it to a reference ground truth string. Precision is the fraction of relevant tokens among the
        retrieved ones (i.e., words in the generated response that also appear in the ground truth), while recall
        is the fraction of relevant tokens that were successfully retrieved (i.e., words in the ground truth that
        are also found in the generated response).

        The F1 score ranges from 0.0 to 1.0, with 1.0 indicating a perfect match. This metric is especially useful
        in tasks where both completeness (recall) and correctness (precision) matter — such as question answering,
        summarization, and any natural language generation task that involves comparison to reference text.

        Attributes:
            response (str): The model-generated response to evaluate.
            ground_truth (str): The correct answer or reference string.
            threshold (float): The F1 score threshold to determine a pass/fail outcome. Must be between 0 and 1.
        """

    def __init__(self, response: str, ground_truth: str, threshold: float):
        super().__init__(response, ground_truth, threshold)

    def get_evaluator(self):
        return F1ScoreEvaluator(threshold=self.threshold)

    def get_result_key(self) -> str:
        return "f1_result"


class RunBleuScoreEvaluator(BaseScoreEvaluator):
    """
       Evaluation Class: Similarity
        Evaluation Method: n-gram
        Granularity: Low

        Uses BLEU (Bilingual Evaluation Understudy) to measure similarity based on n-gram overlap—sequences
        of one or more words—between generated and reference text. Suitable for NLP tasks like machine translation, summarization,
        or text generation where matching short word sequences is a strong indicator of quality.

        Attributes:
            response (str): The generated text.
            ground_truth (str): The reference text.
            threshold (float): BLEU score threshold (0.0 to 1.0).
        """

    def __init__(self, response: str, ground_truth: str, threshold: float):
        super().__init__(response, ground_truth, threshold)

    def get_evaluator(self):
        return BleuScoreEvaluator(threshold=self.threshold)

    def get_result_key(self) -> str:
        return "bleu_result"


class RunGleuScoreEvaluator(BaseScoreEvaluator):
    """
        Evaluation Class: Similarity
        Evaluation Method: n-gram
        Granularity: Low

        Uses GLEU (Google-BLEU) to measure similarity between generated and reference text
        based on n-gram overlap. Unlike BLEU, GLEU accounts for both precision and recall,
        making it more balanced for sentence-level evaluation. Suitable for tasks like
        machine translation, summarization, or text generation where both coverage and conciseness matter.

        Attributes:
            response (str): The generated text.
            ground_truth (str): The reference text.
            threshold (float): GLEU score threshold (0.0 to 1.0).
        """

    def __init__(self, response: str, ground_truth: str, threshold: float):
        super().__init__(response, ground_truth, threshold)

    def get_evaluator(self):
        return GleuScoreEvaluator(threshold=self.threshold)

    def get_result_key(self) -> str:
        return "gleu_result"


class RunRougeScoreEvaluator(BaseScoreEvaluator):
    """
       Evaluation Class: Similarity
       Evaluation Method: n-gram
       Granularity: Low

       Uses ROUGE (Recall-Oriented Understudy for Gisting Evaluation) to assess similarity between
       generated and reference text using ROUGE-L (longest common
       subsequence). Computes F1 score for robust evaluation of summary or
       generation quality.

       Suitable for tasks like summarization and translation where coverage of key information matters.

       Attributes:
           response (str): The generated text.
           ground_truth (str): The reference text.
           threshold (float): Threshold for F1 score (0.0 to 1.0).
       """

    def __init__(self, response: str, ground_truth: str, threshold: float):
        super().__init__(response, ground_truth, threshold)

    def get_evaluator(self):
        return RougeScoreEvaluator(rouge_type=RougeType.ROUGE_L, precision_threshold=self.threshold,
                                   recall_threshold=self.threshold,
                                   f1_score_threshold=self.threshold)

    def get_result_key(self) -> str:
        return "rouge_f1_score_result"


class RunMeteorScoreEvaluator(BaseScoreEvaluator):
    """
        Evaluation Class: Similarity
        Evaluation Method: n-gram + semantic
        Granularity: Low-Medium

        Uses METEOR (Metric for Evaluation of Translation with Explicit Ordering) to assess similarity
        between generated and reference text. Unlike BLEU or ROUGE, METEOR incorporates stemming,
        synonym matching, and word order, making it more semantically aware.

        Suitable for tasks like translation, summarization, and paraphrase detection where both lexical
        and linguistic alignment matter.

        Attributes:
            response (str): The generated text.
            ground_truth (str): The reference text.
            threshold (float): METEOR score threshold (0.0 to 1.0).
        """

    def __init__(self, response: str, ground_truth: str, threshold: float):
        super().__init__(response, ground_truth, threshold)

    def get_evaluator(self):
        return MeteorScoreEvaluator(threshold=self.threshold)

    def get_result_key(self) -> str:
        return "meteor_result"


class RunSemanticSimilarity(RagasBaseEvaluator):
    """
       Evaluation Class: Similarity
       Evaluation Method: Embedding/Cosine Similarity
       Granularity: Medium

       A wrapper class for evaluating the semantic similarity between a model-generated response
       and a predefined ground truth using sentence-level embeddings. Any embedding model can be used in a
       LangchainEmbeddingsWrapper

       The evaluator is useful for:
       - Assessing how well a model's response semantically matches a reference answer.
       - Benchmarking model performance on tasks like question-answering, summarization, and dialogue generation.

       Attributes:
           response (str): The response generated by the model.
           reference (str): The expected correct response (ground truth).
           threshold (float): The minimum similarity score between 0.0 and 1.0.
           embedding_model (LangchainEmbeddingsWrapper): Optional embedding model to calculate the similarity score.
               If not provided, a default Azure OpenAI embedding model will be used.
           ragas_metric_args (dict): Optional arguments for configuring the RAGAS metric.
    """

    def __init__(self, response: str, reference: str, threshold: float,
                 embedding_model: LangchainEmbeddingsWrapper = None):
        embedding_model = embedding_model or get_ragas_wrapped_azure_open_ai_embedding_model()
        super().__init__(sample_data={"response": response, "reference": reference}, threshold=threshold,
                         ragas_metric=SemanticSimilarity, ragas_metric_args={"embeddings": embedding_model})


class RunNonLLMStringSimilarity(RagasBaseEvaluator):  # TODO add distance measures param
    """
        Evaluation Class: RunNonLLMStringSimilarity
        Evaluation Method: String Distance
        Granularity: Low

        This class is used to run the Non-LLM string similarity evaluation, which calculates the string similarity
        between a model-generated response and a reference string by measuring string distances. The evaluation is based
        on distance measures like Levenshtein, Hamming, Jaro, or Jaro-Winkler.

        Attributes:
            response (str): The model-generated response to evaluate.
            reference (str): The reference string against which the model's response will be compared.
            threshold (float): The minimum score required for passing the evaluation, based on the distance measure.
            """

    def __init__(self, response: str, reference: str, threshold: float):
        super().__init__(sample_data={"response": response, "reference": reference}, threshold=threshold,
                         ragas_metric=NonLLMStringSimilarity)


class RunStringPresence(RagasBaseEvaluator):
    """
        Evaluation Class: Similarity
        Evaluation Method: String
        Granularity: Low

        This class is used to evaluate whether a reference string is present within a model-generated response.
        The evaluation is binary — it checks if the entire reference string appears in the response, returning a score
        of 1.0 if the reference string is present, and 0.0 if it is not.

        Attributes:
            response (str): The model-generated response to evaluate.
            reference (str): The reference string to check for presence in the model's response.
            """

    def __init__(self, response: str, reference: str):
        super().__init__(sample_data={"response": response, "reference": reference}, threshold=False,
                         ragas_metric=StringPresence)


class RunExactMatch(RagasBaseEvaluator):
    """
        Evaluation Class: Similarity
        Evaluation Method: Srting
        Granularity: Low

        This class evaluates whether a model-generated response exactly matches a reference string. The evaluation
        is binary — it checks if the entire response string is identical to the reference string, returning a score
        of 1.0 if they are an exact match and 0.0 if they are not.

        Attributes:
            response (str): The model-generated response to evaluate.
            reference (str): The reference string that the model's response is compared against.
            """

    def __init__(self, response: str, reference: str):
        super().__init__(sample_data={"response": response, "reference": reference}, threshold=False,
                         ragas_metric=ExactMatch)
