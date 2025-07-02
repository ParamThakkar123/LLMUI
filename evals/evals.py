import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score

def bleu_score(prediction, ground_truth):
    smoothie = SmoothingFunction().method4
    reference = [nltk.word_tokenize(ground_truth)]
    hypothesis = nltk.word_tokenize(prediction)
    return sentence_bleu(reference, hypothesis, smoothing_function=smoothie)

def batch_bleu(predictions, ground_truth):
    scores = [bleu_score(pred, gt) for pred, gt in zip(predictions, ground_truth)]
    return sum(scores) / len(scores) if scores else 0.0

def bert_score(predictions, ground_truth, lang="en"):
    P, R, F1 = score(predictions, ground_truth, lang=lang, verbose=True)
    return {
        "BERTScore_precision": float(P.mean()),
        "BERTScore_Recall": float(R.mean()),
        "BERTScore_F1": float(F1.mean())
    }