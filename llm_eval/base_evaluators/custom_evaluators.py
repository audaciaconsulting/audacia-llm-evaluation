from llm_eval.tools.model_tools import REQUIRED_MODELS
import re
from enum import Enum

try:
    from transformers import AutoTokenizer, pipeline
except ImportError as e:
    raise ImportError(
        "The sentiment, toxicity and bias evaluators require the optional "
        "'local-models' dependencies (torch, transformers). Install them with: "
        "`uv sync --extra local-models` (in a clone) or "
        "`uv add \"audacia-llm-evaluation[local-models]\"` (as a dependency)."
    ) from e

class AggregationStrategy(Enum):
    """Strategy for aggregating scores across text chunks."""
    FULL_CONTEXT = "full_context"
    MIN_SENTENCE_SCORE = "min_sentence_score"
    MAX_SENTENCE_SCORE = "max_sentence_score"


class TransformerEvaluator:
    """
    A general-purpose evaluator for text classification using Hugging Face Transformers.

    This class wraps a classification pipeline and allows for either single-label or weighted aggregate
    scoring, depending on initialization parameters. Supports different aggregation strategies for
    processing text at different granularities.

    Args:
        evaluator (str): Key to retrieve the model name from REQUIRED_MODELS.
        label_index (int, optional): Index of the label to extract the score from if not aggregating. Defaults to 0.
        aggregate (bool, optional): Whether to compute a weighted aggregate score across all labels. Defaults to False.
        aggregate_weights (dict, optional): Dictionary of label weights used during aggregation. Required if aggregate is True.
        max_length (int, optional): Maximum token length for model input. Defaults to 512.
        overlap_sentences (int, optional): Number of sentences to overlap between chunks. Defaults to 1.
        aggregation_strategy (AggregationStrategy, optional): Strategy for aggregating scores. Defaults to FULL_CONTEXT.

    Example:
        evaluator = TransformerEvaluator(
            "sentiment",
            aggregate=True,
            aggregate_weights=...,
            aggregation_strategy=AggregationStrategy.MIN_SENTENCE_SCORE
        )
        result = evaluator(response="The response text.")
    """

    def __init__(
            self,
            evaluator: str,
            *,
            label_index: int = 0,
            aggregate: bool = False,
            aggregate_weights: dict = None,
            max_length: int = 512,
            overlap_sentences: int = 1,
            aggregation_strategy: AggregationStrategy = AggregationStrategy.FULL_CONTEXT,
    ):
        self.evaluator = evaluator
        self.label_index = label_index
        self.aggregate = aggregate
        self.aggregate_weights = aggregate_weights
        self.max_length = max_length
        self.overlap_sentences = overlap_sentences
        self.aggregation_strategy = aggregation_strategy

        # Initialize tokenizer and classifier once
        model_name = REQUIRED_MODELS[self.evaluator]["name"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.classifier = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=self.tokenizer,
            return_all_scores=True,
            device="cpu",
            truncation=True,
            max_length=self.max_length,
        )

    def _split_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences using regex.

        Args:
            text (str): Input text to split.

        Returns:
            list[str]: List of sentences.
        """
        # Split on sentence boundaries (., !, ?) followed by whitespace or end of string
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _chunk_sentences_with_overlap(self, sentences: list[str]) -> list[str]:
        """
        Group sentences into chunks that fit within max_length with overlapping sentences.

        Args:
            sentences (list[str]): List of sentences.

        Returns:
            list[str]: List of text chunks with overlap.
        """
        if not sentences:
            return []

        chunks = []
        current_chunk = []
        current_length = 0
        effective_max = self.max_length - 2  # Reserve space for special tokens
        i = 0

        while i < len(sentences):
            sentence = sentences[i]
            sentence_tokens = self.tokenizer.encode(sentence, add_special_tokens=False)
            sentence_length = len(sentence_tokens)

            # If single sentence exceeds max_length, add it as its own chunk (will be truncated)
            if sentence_length > effective_max:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                chunks.append(sentence)
                i += 1
                continue

            # If adding this sentence would exceed max_length, start new chunk
            if current_length + sentence_length > effective_max:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))

                    # Create overlap: keep last N sentences for next chunk
                    overlap_size = min(self.overlap_sentences, len(current_chunk))
                    if overlap_size > 0:
                        overlap_sentences = current_chunk[-overlap_size:]
                        overlap_tokens = [
                            self.tokenizer.encode(s, add_special_tokens=False)
                            for s in overlap_sentences
                        ]
                        overlap_length = sum(len(tokens) for tokens in overlap_tokens)

                        current_chunk = overlap_sentences
                        current_length = overlap_length
                    else:
                        current_chunk = []
                        current_length = 0
                else:
                    current_chunk = []
                    current_length = 0
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
                i += 1

        # Add remaining sentences
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _aggregate_chunk_results(self, chunk_results: list[list[dict]]) -> list[dict]:
        """
        Aggregate results from multiple chunks by averaging scores per label.

        Args:
            chunk_results (list[list[dict]]): Results from each chunk.

        Returns:
            list[dict]: Aggregated results with averaged scores.
        """
        if len(chunk_results) == 1:
            return chunk_results[0]

        # Collect scores by label
        label_scores = {}
        for chunk_result in chunk_results:
            for item in chunk_result:
                label = item["label"]
                score = item["score"]
                if label not in label_scores:
                    label_scores[label] = []
                label_scores[label].append(score)

        # Average scores for each label
        aggregated = [
            {"label": label, "score": sum(scores) / len(scores)}
            for label, scores in label_scores.items()
        ]

        return aggregated

    def _extract_score_from_results(self, results: list[dict]) -> float:
        """
        Extract final score from classification results based on aggregate settings.

        Args:
            results (list[dict]): Classification results with labels and scores.

        Returns:
            float: Extracted score.
        """
        if self.aggregate and self.aggregate_weights:
            return sum(
                self.aggregate_weights[x["label"]] * x["score"] for x in results
            )
        else:
            return results[self.label_index]["score"]

    def __call__(self, *, response: str, **kwargs):
        """
        Evaluates the response using the configured text classification model.
        Behavior depends on aggregation_strategy:
        - FULL_CONTEXT: Splits text by sentences, chunks by token limits with overlap, aggregates results.
        - MIN_SENTENCE_SCORE: Scores each sentence individually and returns the minimum score.
        - MAX_SENTENCE_SCORE: Scores each sentence individually and returns the maximum score.

        Args:
            response (str): The textual response to evaluate.
            **kwargs: Additional keyword arguments (ignored in current implementation).

        Returns:
            dict: A dictionary containing the evaluation score with the evaluator name as the key.
        """
        # Split into sentences
        sentences = self._split_sentences(response)

        if self.aggregation_strategy == AggregationStrategy.FULL_CONTEXT:
            # Group sentences into overlapping chunks that fit max_length
            chunks = self._chunk_sentences_with_overlap(sentences)

            # Classify each chunk
            chunk_results = [self.classifier(chunk)[0] for chunk in chunks]

            # Aggregate results across chunks
            results = self._aggregate_chunk_results(chunk_results)

            # Compute final score
            score = self._extract_score_from_results(results)

        elif self.aggregation_strategy in (
            AggregationStrategy.MIN_SENTENCE_SCORE,
            AggregationStrategy.MAX_SENTENCE_SCORE
        ):
            # Score each sentence individually
            sentence_scores = []
            for sentence in sentences:
                results = self.classifier(sentence)[0]
                sentence_score = self._extract_score_from_results(results)
                sentence_scores.append({"sentence": sentence, "score": sentence_score})

            # Return min or max score
            if self.aggregation_strategy == AggregationStrategy.MIN_SENTENCE_SCORE:
                selected = (
                    min(sentence_scores, key=lambda x: x["score"])
                    if sentence_scores
                    else {"sentence": None, "score": 0.0}
                )
                score = selected["score"]
                min_sentence = selected["sentence"]
                return {
                    self.evaluator: score,
                    "min_sentence": min_sentence,
                }
            else:  # MAX_SENTENCE_SCORE
                selected = (
                    max(sentence_scores, key=lambda x: x["score"])
                    if sentence_scores
                    else {"sentence": None, "score": 0.0}
                )
                score = selected["score"]
                max_sentence = selected["sentence"]
                return {
                    self.evaluator: score,
                    "max_sentence": max_sentence,
                }

        return {self.evaluator: score}


class SentimentEvaluator(TransformerEvaluator):
    """
    Evaluates the sentiment of a response using a predefined transformer model.

    Maps sentiment labels to numerical values using a predefined weighting scheme and computes
    an aggregate sentiment score.

    Scoring weights:
        - "Very Negative": -1.0
        - "Negative": -0.5
        - "Neutral": 0.0
        - "Positive": 0.5
        - "Very Positive": 1.0

    Example:
        evaluator = SentimentEvaluator()
        result = evaluator(response="This is a great product!")
    """

    def __init__(self, aggregation_strategy: AggregationStrategy = AggregationStrategy.FULL_CONTEXT):
        WEIGHTS = {
            "Very Negative": -1.0,
            "Negative": -0.5,
            "Neutral": 0.0,
            "Positive": 0.5,
            "Very Positive": 1.0,
        }
        super().__init__(
            evaluator="sentiment",
            aggregate=True,
            aggregate_weights=WEIGHTS,
            aggregation_strategy=aggregation_strategy,
        )


class BiasEvaluator(TransformerEvaluator):
    """
    Evaluates the bias score of a response using a transformer model.

    Selects the score from a specific label index (default 0), which is assumed
    to represent the target bias class.

    Example:
        evaluator = BiasEvaluator()
        result = evaluator(response="That’s not how everyone sees it.")
    """

    def __init__(self, aggregation_strategy: AggregationStrategy = AggregationStrategy.FULL_CONTEXT):
        super().__init__(evaluator="bias", label_index=0, aggregation_strategy=aggregation_strategy)


class ToxicityEvaluator(TransformerEvaluator):
    """
    Evaluates the toxicity of a response using a transformer model.

    Selects the score from a specific label index (default 1), which is assumed
    to correspond to the toxicity class in the classification output.

    Example:
        evaluator = ToxicityEvaluator()
        result = evaluator(response="You’re an idiot.")
    """

    def __init__(self, aggregation_strategy: AggregationStrategy = AggregationStrategy.FULL_CONTEXT):
        super().__init__(evaluator="toxicity", label_index=1, aggregation_strategy=aggregation_strategy)
