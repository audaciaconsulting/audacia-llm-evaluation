from typing import Any
from transformers import pipeline

from llm_eval.tools.model_tools import REQUIRED_MODELS


class TransformerEvaluator:
    """
    A general-purpose evaluator for text classification using Hugging Face Transformers.

    This class wraps a classification pipeline and allows for either single-label or weighted aggregate
    scoring, depending on initialization parameters.

    Args:
        evaluator (str): Key to retrieve the model name from REQUIRED_MODELS.
        label_index (int, optional): Index of the label to extract the score from if not aggregating. Defaults to 0.
        aggregate (bool, optional): Whether to compute a weighted aggregate score across all labels. Defaults to False.
        aggregate_weights (dict, optional): Dictionary of label weights used during aggregation. Required if aggregate is True.

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
    ):
        self.evaluator = evaluator
        self.label_index = label_index
        self.aggregate = aggregate
        self.aggregate_weights = aggregate_weights

    def __call__(self, *, response: str, **kwargs):
        """
        Evaluates the response using the configured text classification model.

        Args:
            response (str): The textual response to evaluate.
            **kwargs: Additional keyword arguments (ignored in current implementation).

        Returns:
            dict: A dictionary containing the evaluation score with the evaluator name as the key.
        """
        classifier = pipeline(
            "text-classification",
            model=REQUIRED_MODELS[self.evaluator]["name"],
            return_all_scores=True,
            device="cpu",
        )
        results = classifier(response)[0]

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


class FormatEvaluator:
    """Evaluator that inspects the format type of a given response object.
    This class provides a callable interface to determine the type of the
    input response. It returns a dictionary with the format information,
    useful for downstream format validation or logging purposes.
    Methods:
        __call__(response: Any, **kwargs): Evaluates and returns the type of the response.
    Example:
        evaluator = FormatEvaluator()
        result = evaluator(response="hello")
        # result -> {'format': <class 'str'>}
    """

    def __init__(self):
        pass

    def __call__(self, *, response: Any, **kwargs):
        """Evaluates the format (type) of the response.
        Args:
            response (Any): The response object to evaluate.
            **kwargs: Additional keyword arguments (ignored by default).
        Returns:
            dict: A dictionary containing the type of the response under the key 'format'.
        """
        return {"format": type(response)}
