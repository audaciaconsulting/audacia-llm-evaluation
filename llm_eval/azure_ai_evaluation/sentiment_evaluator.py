import logging
from statistics import mean, stdev
from typing import List

from llm_eval.custom_evaluator_tools import SentimentEvaluator

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def format_dict_log(dictionary: dict):
    lines = '\n'.join(f"{k}: {v}" for k, v in dictionary.items())
    return f"\n\n{'*'*100}\n\n{lines}\n\n{'*'*100}\n\n"

class RunSentimentEvaluator:
    def __init__(self, response: str):
        self.response = response

    def __call__(self):
        evaluator = SentimentEvaluator()
        score = evaluator(
            response=self.response,
        )
        return score

    def evaluate_against_expected_score(
        self, expected_score: float, allowed_uncertainty: float = 0.05
    ):
        result = self()
        pass_state = (
            expected_score - allowed_uncertainty
            < result["sentiment"]
            < expected_score + allowed_uncertainty
        )
        result.update(
            {
                "response": self.response,
                "expected_score": expected_score,
                "evaluation_result": pass_state,
            }
        )
        logger.info(format_dict_log(dictionary=result))
        assert pass_state

    def evaluate_against_golden_standards(
        self, golden_standards: List[str], scale_uncertainty: int = 1
    ):
        current_result = self()
        golden_standard_scores = []
        for response in golden_standards:
            evaluator = SentimentEvaluator()
            golden_standard_scores.append(evaluator(response=response)["sentiment"])
        score_mean = mean(golden_standard_scores)
        score_uncertainty = stdev(golden_standard_scores) * scale_uncertainty
        pass_state = (
            score_mean - score_uncertainty
            < current_result["sentiment"]
            < score_mean + score_uncertainty
        )
        current_result.update(
            {
                "response": self.response,
                "golden_standard_example": golden_standard_scores[0],
                "golden_standard_score": score_mean,
                "calculated_uncertainty": score_uncertainty,
                "evaluation_result": pass_state,
            }
        )
        logger.info(format_dict_log(dictionary=current_result))
        assert pass_state
