from llm_eval.tools.model_tools import REQUIRED_MODELS
import re
from transformers import AutoTokenizer, pipeline


class TransformerEvaluator:
    """
    A general-purpose evaluator for text classification using Hugging Face Transformers.

    This class wraps a classification pipeline and allows for either single-label or weighted aggregate
    scoring, depending on initialization parameters. Always chunks text by sentences and aggregates results.

    Args:
        evaluator (str): Key to retrieve the model name from REQUIRED_MODELS.
        label_index (int, optional): Index of the label to extract the score from if not aggregating. Defaults to 0.
        aggregate (bool, optional): Whether to compute a weighted aggregate score across all labels. Defaults to False.
        aggregate_weights (dict, optional): Dictionary of label weights used during aggregation. Required if aggregate is True.
        max_length (int, optional): Maximum token length for model input. Defaults to 512.
        overlap_sentences (int, optional): Number of sentences to overlap between chunks. Defaults to 1.

    Example:
        evaluator = TransformerEvaluator("sentiment", aggregate=True, aggregate_weights=...)
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
    ):
        self.evaluator = evaluator
        self.label_index = label_index
        self.aggregate = aggregate
        self.aggregate_weights = aggregate_weights
        self.max_length = max_length
        self.overlap_sentences = overlap_sentences

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

    def __call__(self, *, response: str, **kwargs):
        """
        Evaluates the response using the configured text classification model.
        Splits text by sentences, chunks by token limits with overlap, and aggregates results.

        Args:
            response (str): The textual response to evaluate.
            **kwargs: Additional keyword arguments (ignored in current implementation).

        Returns:
            dict: A dictionary containing the evaluation score with the evaluator name as the key.
        """
        # Split into sentences
        sentences = self._split_sentences(response)

        # Group sentences into overlapping chunks that fit max_length
        chunks = self._chunk_sentences_with_overlap(sentences)

        # Classify each chunk
        chunk_results = [self.classifier(chunk)[0] for chunk in chunks]

        # Aggregate results across chunks
        results = self._aggregate_chunk_results(chunk_results)

        # Compute final score
        if self.aggregate and self.aggregate_weights:
            score = sum(
                self.aggregate_weights[x["label"]] * x["score"] for x in results
            )
        else:
            score = results[self.label_index]["score"]

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

    def __init__(self):
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

    def __init__(self):
        super().__init__(evaluator="bias", label_index=0)


class ToxicityEvaluator(TransformerEvaluator):
    """
    Evaluates the toxicity of a response using a transformer model.

    Selects the score from a specific label index (default 1), which is assumed
    to correspond to the toxicity class in the classification output.

    Example:
        evaluator = ToxicityEvaluator()
        result = evaluator(response="You’re an idiot.")
    """

    def __init__(self):
        super().__init__(evaluator="toxicity", label_index=1)
