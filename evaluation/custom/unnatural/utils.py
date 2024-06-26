import collections
import re
import string

import evaluate


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(the|a|an)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


# Exact match (the normalized answer exactly match the gold answer)
def exact(predictions, references):
    return int(normalize_answer(references[0]) == normalize_answer(predictions[0]))


# The F-score of predicted tokens versus the gold answer
def f1(predictions, references):
    gold_toks = get_tokens(references[0])
    pred_toks = get_tokens(predictions[0])
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def rouge1(items):
    """
    # passthrough for efficiency
    """
    return items


def rouge1_agg(items):
    """
    Higher is better
    """
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    rouge_scorer = evaluate.load("rouge")
    return rouge_scorer.compute(predictions=preds, references=refs)["rouge1"]

def doc_to_text(record):
    instances = record["instances"]
    instruction = record["instruction"]
    input = instances[0]["input"]
    

    return f"{instruction} ###{input} ###Response:"

def doc_to_target(record):
    instances = record["instances"]
    doc = instances[0]

    return doc["output"]