"""Explicit denominators. ID overlap and regex alerts are not semantic judgments."""
import re


def ratio(numerator, denominator):
    if type(numerator) is not int or type(denominator) is not int or not 0 <= numerator <= denominator:
        raise ValueError('Metric counts must satisfy 0 <= numerator <= denominator')
    return numerator / denominator if denominator else None


def compute_alr(answer_text=None, test_case=None, *, leaked_assertions=None, eligible_assertions=None):
    """Authoritative ALR needs reviewed counts; raw text alone is insufficient."""
    if leaked_assertions is None or eligible_assertions is None:
        return None
    return ratio(leaked_assertions, eligible_assertions)


def leakage_alerts(answer_text, test_case):
    """Diagnostic regex hits only; negation/comparisons can create false positives."""
    alerts = []
    for index, rule in enumerate(test_case.get('leakage_rules', [])):
        pattern = re.escape(rule['subject']) + r'[^.\n]*?' + re.escape(rule['forbidden_value'])
        for match in re.finditer(pattern, answer_text, re.IGNORECASE):
            alerts.append(dict(rule_index=index, text=match.group(), span=list(match.span()),
                               status='needs_semantic_review'))
    return alerts


def citation_id_overlap(cited_ids, gold_ids):
    cited, gold = set(cited_ids), set(gold_ids)
    if not gold:
        return dict(precision=None, recall=None, f1=None)
    hits = len(cited & gold)
    precision = hits / len(cited) if cited else 0.0
    recall = hits / len(gold)
    return dict(precision=precision, recall=recall,
                f1=2*precision*recall/(precision+recall) if precision+recall else 0.0)


def evaluate_citation_faithfulness(*, supported_cited_claims=None, cited_claims=None):
    if supported_cited_claims is None or cited_claims is None:
        return None
    return ratio(supported_cited_claims, cited_claims)


def evidence_metrics(selected, gold, required, k):
    if k < 1:
        raise ValueError('k must be positive')
    top = list(dict.fromkeys(selected))[:k]
    return dict(evidence_recall_at_k=len(set(top)&set(gold))/len(set(gold)) if gold else None,
        evidence_precision_at_k=len(set(top)&set(gold))/len(top) if gold and top else (0.0 if gold else None),
        required_evidence_recall=len(set(selected)&set(required))/len(set(required)) if required else None,
        precision_denominator='returned_unique_evidence_up_to_k',
        required_evidence_recall_scope='retrieved_or_selected_ids; does not prove condition interpretation')


def review_metrics(review):
    if review.get('status') != 'approved' or not review.get('reviewer'):
        return {}
    accuracy = review.get('answer_accuracy')
    if type(accuracy) is not int or accuracy not in (0,1):
        raise ValueError('Reviewed answer_accuracy must be 0 or 1')
    return dict(accuracy=accuracy,
        attribute_leakage_rate=ratio(review['leaked_assertions'], review['eligible_assertions']),
        citation_faithfulness=ratio(review['supported_cited_claims'], review['cited_claims']))
