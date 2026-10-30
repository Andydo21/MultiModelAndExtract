"""Reference-based metrics; scores require labels, never retrieval confidence."""
import math
import unicodedata


def normalized_text(value):
    return ' '.join(unicodedata.normalize('NFC', str(value)).split())


def edit_distance(reference, prediction):
    previous = list(range(len(prediction) + 1))
    for i, expected in enumerate(reference, 1):
        current = [i]
        for j, actual in enumerate(prediction, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (expected != actual)))
        previous = current
    return previous[-1]


def ocr_counts(reference, prediction):
    reference, prediction = normalized_text(reference), normalized_text(prediction)
    return dict(character_errors=edit_distance(reference, prediction),
                reference_characters=len(reference),
                word_errors=edit_distance(reference.split(), prediction.split()),
                reference_words=len(reference.split()))


def ranking_metrics(ranked_ids, relevant_ids, k):
    if k < 1:
        raise ValueError('k must be positive')
    relevant = set(relevant_ids)
    if not relevant:
        return dict(recall=None, reciprocal_rank=None, ndcg=None)
    ranked = list(dict.fromkeys(ranked_ids))[:k]
    hits = [int(identifier in relevant) for identifier in ranked]
    dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(relevant))))
    return dict(recall=sum(hits) / len(relevant),
                reciprocal_rank=next((1 / (i + 1) for i, hit in enumerate(hits) if hit), 0),
                ndcg=dcg / ideal)


def field_metrics(expected, predicted):
    """Micro P/R/F1 on exact values; a wrong filled field is both FP and FN."""
    rows, tp, fp, fn = [], 0, 0, 0
    for key in sorted(set(expected) | set(predicted)):
        gold = normalized_text(expected.get(key, ''))
        actual = normalized_text(predicted.get(key, ''))
        correct = gold == actual
        tp += bool(gold and correct)
        fp += bool(actual and not correct)
        fn += bool(gold and not correct)
        rows.append(dict(field=key, expected=gold, predicted=actual, exact_match=correct))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return dict(rows=rows, tp=tp, fp=fp, fn=fn, precision=precision, recall=recall,
                f1=f1, exact_match=sum(r['exact_match'] for r in rows) / len(rows) if rows else None)
