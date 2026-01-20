import re
import string
import unicodedata
from collections import Counter
import asyncio


def normalize_text(s):
    s = unicodedata.normalize('NFD', s)

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


class BaseMetric:
    """`BaseMetric` serves as the base object of all metrics. Implemented metric should
    inherit this class.
    """

    metric_name = "base"

    def __init__(self):
        ...

    def cal_one(self, response: str, answer: str):
        raise NotImplementedError

    def cal(self, response: str, golden_answers: list[str]):
        scores = [self.cal_one(response, golden_answer) for golden_answer in golden_answers]
        return max(scores)

    async def cal_one(self, question: str, response: str, answer: str, path):
        raise NotImplementedError

    async def cal(self, question: str, response: str, golden_answers: list[str], path):
        task = [self.cal_one(question, response, golden_answer, path) for golden_answer in golden_answers]
        scores = await asyncio.gather(*task)
        return max(scores)


class F1_Score(BaseMetric):
    """Token-level F1 score"""

    metric_name = ["f1", "precision", "recall"]

    def cal(self, response: str, golden_answers: list[str]):
        final_metric = {"f1": 0, "precision": 0, "recall": 0}
        if isinstance(golden_answers, str):
            golden_answers = [golden_answers]
        for ground_truth in golden_answers:
            normalized_prediction = normalize_text(response)
            normalized_ground_truth = normalize_text(ground_truth)
            prediction_tokens = normalized_prediction.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            for k in ["f1", "precision", "recall"]:
                final_metric[k] = max(eval(k), final_metric[k])
        return final_metric


class ExactMatch(BaseMetric):
    r"""Exact match measure whether the predicted answer is completely consistent
    with the standard answer.

    """

    metric_name = "em"

    def cal(self, response: str, golden_answers: list[str]) -> float:
        if isinstance(golden_answers, str):
            golden_answers = [golden_answers]
        normalized_prediction = normalize_text(response)
        score = 0.0
        for golden_answer in golden_answers:
            golden_answer = normalize_text(golden_answer)
            if golden_answer == normalized_prediction:
                score = 1.0
                break
        return score


class Sub_ExactMatch(BaseMetric):
    r"""Sub-Exact match measure whether the predicted answer contains the standard answer."""

    metric_name = "acc"

    def cal(self, response: str, golden_answers: list[str]) -> float:
        if isinstance(golden_answers, str):
            golden_answers = [golden_answers]
        normalized_prediction = normalize_text(response)
        score = 0.0
        for golden_answer in golden_answers:
            golden_answer = normalize_text(golden_answer)
            if golden_answer in normalized_prediction:
                score = 1.0
                break
        return score
