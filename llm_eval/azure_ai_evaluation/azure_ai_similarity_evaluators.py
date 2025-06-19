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

from llm_eval.azure_ai_evaluation.azure_ai_similarity_base_evaluator import BaseScoreEvaluator
from llm_eval.model_tools import get_azure_ai_evaluation_model_config
import logging

from utils import format_dict_log

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

    def evaluate(self):
        result = self()

        result.update(
            {
                "query": self.query,
                "response": self.response,
                "ground_truth": self.ground_truth
            }
        )

        logger.info(format_dict_log(dictionary=result))
        assert result['similarity_result'] == 'pass'


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

    def get_evaluator(self):
        return MeteorScoreEvaluator(threshold=self.threshold)

    def get_result_key(self) -> str:
        return "meteor_result"
