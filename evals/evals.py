from bert_score import score
from rouge_score import rouge_scorer
from collections import Counter

def simple_tokenize(text):
    import re
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

def bleu_score(prediction, ground_truth, smoothing=True):
    reference = [simple_tokenize(ground_truth)]
    hypothesis = simple_tokenize(prediction)
    overlap = Counter(hypothesis) & Counter(reference[0])
    overlap_count = sum(overlap.values())
    if smoothing:
        precision = (overlap_count + 1) / (len(hypothesis) + 1) if hypothesis else 0.0
    else:
        precision = overlap_count / len(hypothesis) if hypothesis else 0.0
    return precision

def batch_bleu(predictions, ground_truth, smoothing=True):
    scores = [bleu_score(pred, gt, smoothing=smoothing) for pred, gt in zip(predictions, ground_truth)]
    return sum(scores) / len(scores) if scores else 0.0

def bert_score(predictions, ground_truth, lang="en"):
    P, R, F1 = score(predictions, ground_truth, lang=lang, verbose=True)
    return {
        "BERTScore_precision": float(P.mean()),
        "BERTScore_Recall": float(R.mean()),
        "BERTScore_F1": float(F1.mean())
    }

def rouge_l(predictions, ground_truth):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = [scorer.score(gt, pred)['rougeL'].fmeasure for pred, gt in zip(predictions, ground_truth)]
    return sum(scores) / len(scores) if scores else 0.0

def exact_match(predictions, ground_truth):
    matches = [int(pred.strip() == gt.strip()) for pred, gt in zip(predictions, ground_truth)]
    return sum(matches) / len(matches) if matches else 0.0

def f1(predictions, ground_truth):
    f1s = []
    for pred, gt in zip(predictions, ground_truth):
        pred_tokens = simple_tokenize(pred)
        gt_tokens = simple_tokenize(gt)
        common = set(pred_tokens) & set(gt_tokens)
        if not pred_tokens or not gt_tokens:
            f1s.append(0.0)
            continue
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gt_tokens)
        if precision + recall == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * precision * recall / (precision + recall))
    return sum(f1s) / len(f1s) if f1s else 0.0

def precision_at_k(predictions, ground_truth, k=5):
    precisions = []
    for pred_list, gt in zip(predictions, ground_truth):
        pred_set = set(pred_list[:k])
        gt_set = set([gt])
        precisions.append(len(pred_set & gt_set) / k)
    return sum(precisions) / len(precisions) if precisions else 0.0

def recall_at_k(predictions, ground_truth, k=5):
    recalls = []
    for pred_list, gt in zip(predictions, ground_truth):
        pred_set = set(pred_list[:k])
        gt_set = set([gt])
        recalls.append(len(pred_set & gt_set) / len(gt_set) if gt_set else 0.0)
    return sum(recalls) / len(recalls) if recalls else 0.0