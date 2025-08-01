import logging
from typing import Optional

from azure.ai.evaluation import (
    AzureOpenAIModelConfiguration,
    BleuScoreEvaluator,
    F1ScoreEvaluator,
    GleuScoreEvaluator,
    MeteorScoreEvaluator,
    RougeScoreEvaluator,
    RougeType,
    SimilarityEvaluator,
)
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    ExactMatch,
    NonLLMStringSimilarity,
    SemanticSimilarity,
    StringPresence,
)

from llm_eval.base_evaluators.azure_ai_similarity_base_evaluator import (
    BaseScoreEvaluator,
)
from llm_eval.base_evaluators.ragas_base_evaluator import RagasBaseEvaluator
from llm_eval.tools.model_tools import (
    get_azure_ai_evaluation_model_config,
    get_ragas_wrapped_azure_open_ai_embedding_model,
)
from llm_eval.tools.utils import format_dict_log

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RunSimilarityEvaluator:
    """
    Evaluation Class: Similarity  
    Evaluation Method: LLM-Based Prompt Evaluation  
    Granularity: High

    The similarity score is computed using a large language model (LLM) via prompt-based evaluation. Instead of using
    embedding-based cosine similarity, this method presents the query, model-generated response, and ground truth to
    an LLM using a structured prompt defined in `similarity.prompty`. The model then evaluates how semantically
    aligned the response is with the ground truth, producing a score between 1.0 (least similar) and 5.0 (most similar).

    The `query` provides context for the comparison, helping the LLM judge whether the response matches the intent and
    meaning of the ground truth in context.

    This evaluation is especially useful in tasks such as question answering, summarization, and other natural language
    generation settings where semantic correctness matters. Because it leverages an LLM's judgment, it enables more
    context-aware and nuanced similarity scoring than purely embedding-based approaches.

    Attributes:
        query (str): The input query or prompt providing context for comparison.
        response (str): The model-generated response to evaluate.
        reference (str): The expected correct response or reference text.
        threshold (float): The minimum similarity score required to pass the evaluation, ranging from 1.0 to 5.0.
        model_config (Optional[AzureOpenAIModelConfiguration]): Configuration for the LLM used in evaluation.
    """


    def __init__(
        self,
        query: str,
        response: str,
        reference: str,
        threshold: float,
        model_config: Optional[AzureOpenAIModelConfiguration],
    ):
        self.query = query
        self.response = response
        self.reference = reference
        self.threshold = threshold
        self.model_config = model_config or get_azure_ai_evaluation_model_config()

        if not 0.0 <= threshold <= 5.0:
            raise ValueError(f"Threshold must be between 0 and 5. Got {threshold}.")

    def __call__(self) -> dict:
        evaluator = SimilarityEvaluator(
            model_config=self.model_config, threshold=self.threshold
        )
        return evaluator(
            query=self.query,
            response=self.response,
            ground_truth=self.reference,
        )

    def assert_result(self):
        result = self()
        if result.get("similarity_result") == "fail":
            raise AssertionError("Similarity evaluation failed against ground truth")

    def evaluate(self, assert_result: bool = False):
        result = self()

        result.update(
            {
                "query": self.query,
                "response": self.response,
                "reference": self.reference,
            }
        )

        logger.info(format_dict_log(dictionary=result))

        if assert_result:
            assert result["similarity_result"] == "pass"

        return result


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
        reference (str): The reference text.
        threshold (float): METEOR score threshold (0.0 to 1.0).
    """

    def __init__(self, response: str, reference: str, threshold: float):
        evaluator = MeteorScoreEvaluator(threshold=threshold)
        super().__init__(
            response,
            reference,
            threshold,
            result_key="meteor_result",
            evaluator=evaluator,
            assertion_fail_message="Evaluation failed: the METEOR similarity score is not within the acceptable threshold",
        )


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
         reference (str): The reference text.
         threshold (float): BLEU score threshold (0.0 to 1.0).
    """

    def __init__(self, response: str, reference: str, threshold: float):
        evaluator = BleuScoreEvaluator(threshold=threshold)
        super().__init__(
            response,
            reference,
            threshold,
            result_key="bleu_result",
            evaluator=evaluator,
            assertion_fail_message="Evaluation failed: the BLUE similarity score is not within the acceptable threshold",
        )


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
        reference (str): The reference text.
        threshold (float): GLEU score threshold (0.0 to 1.0).
    """

    def __init__(self, response: str, reference: str, threshold: float):
        evaluator = GleuScoreEvaluator(threshold=threshold)
        super().__init__(
            response,
            reference,
            threshold,
            result_key="gleu_result",
            evaluator=evaluator,
            assertion_fail_message="Evaluation failed: the GLEU similarity score is not within the acceptable threshold",
        )


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
        reference (str): The reference text.
        threshold (float): Threshold for F1 score (0.0 to 1.0).
    """

    def __init__(self, response: str, reference: str, threshold: float):
        evaluator = RougeScoreEvaluator(
            rouge_type=RougeType.ROUGE_L,
            precision_threshold=threshold,
            recall_threshold=threshold,
            f1_score_threshold=threshold,
        )
        super().__init__(
            response,
            reference,
            threshold,
            result_key="rouge_f1_score_result",
            evaluator=evaluator,
            assertion_fail_message="Evaluation failed: the ROUGE similarity score is not within the acceptable threshold",
        )


class RunF1ScoreEvaluator(BaseScoreEvaluator):
    """
    Evaluation Class: Similarity
    Evaluation Method: Words
    Granularity: Low

    The F1 score is a harmonic mean of precision and recall. It measures the accuracy of a generated response
    by comparing it to a reference ground truth string. Precision is the fraction of relevant words among the
    retrieved ones (i.e., words in the generated response that also appear in the ground truth), while recall
    is the fraction of relevant words that were successfully retrieved (i.e., words in the ground truth that
    are also found in the generated response).

    The F1 score ranges from 0.0 to 1.0, with 1.0 indicating a perfect match. This metric is especially useful
    in tasks where both completeness (recall) and correctness (precision) matter — such as question answering,
    summarization, and any natural language generation task that involves comparison to reference text.

    Attributes:
        response (str): The model-generated response to evaluate.
        reference (str): The correct answer or reference string.
        threshold (float): The F1 score threshold to determine a pass/fail outcome. Must be between 0 and 1.
    """

    def __init__(self, response: str, reference: str, threshold: float):
        evaluator = F1ScoreEvaluator(threshold=threshold)
        super().__init__(
            response,
            reference,
            threshold,
            result_key="f1_result",
            evaluator=evaluator,
            assertion_fail_message="Evaluation failed: the F1 similarity score is not within the acceptable threshold",
        )


class RunSemanticSimilarityEvaluator(RagasBaseEvaluator):
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

    def __init__(
        self,
        response: str,
        reference: str,
        threshold: float,
        embedding_model: LangchainEmbeddingsWrapper = None,
    ):
        embedding_model = (
            embedding_model or get_ragas_wrapped_azure_open_ai_embedding_model()
        )
        super().__init__(
            sample_data={"response": response, "reference": reference},
            threshold=threshold,
            ragas_metric=SemanticSimilarity,
            ragas_metric_args={"embeddings": embedding_model},
            assertion_fail_message="Evaluation failed: response too semantically different to the reference using ragas LLM as a judge method",
        )


class RunNonLLMStringSimilarityEvaluator(RagasBaseEvaluator):  # TODO add distance measures param
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
        super().__init__(
            sample_data={"response": response, "reference": reference},
            threshold=threshold,
            ragas_metric=NonLLMStringSimilarity,
            assertion_fail_message="Evaluation failed: response too semantically different to the reference using ragas non-LLM as a judge method",
        )


class RunStringPresenceEvaluator(RagasBaseEvaluator):
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
        super().__init__(
            sample_data={"response": response, "reference": reference},
            threshold=False,
            ragas_metric=StringPresence,
            assertion_fail_message="Evaluation failed: the reference string does not exist within the response",
        )


class RunExactMatchEvaluator(RagasBaseEvaluator):
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
        super().__init__(
            sample_data={"response": response, "reference": reference},
            threshold=False,
            ragas_metric=ExactMatch,
            assertion_fail_message="Evaluation failed: there are differences between the response and the reference.",
        )
