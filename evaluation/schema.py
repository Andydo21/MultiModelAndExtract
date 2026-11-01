"""Source-bound question, fact and run contracts. No generated gold defaults."""
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GroundTruth:
    subject: str
    attribute: str
    value: Any
    condition: str | None
    evidence_ids: list[str]


@dataclass(frozen=True)
class Question:
    question_id: str
    question: str
    split: str
    categories: list[str]
    reference_answer: str
    ground_truth_tuples: list[GroundTruth]
    gold_evidence_ids: list[str]
    required_evidence_ids: list[str]
    evidence_locations: list[dict]
    reviewer: str
    review_status: str


@dataclass
class RunResult:
    experiment: str
    question_id: str
    variant: str
    repetition: int
    status: str = 'pending'
    prompt: list[dict] = field(default_factory=list)
    raw_response: dict | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    accuracy: float | None = None
    attribute_leakage_rate: float | None = None
    citation_faithfulness: float | None = None


def validate_evidence(evidence):
    if not isinstance(evidence, list) or not evidence:
        raise ValueError('A nonempty actual evidence list is required')
    ids = set()
    for item in evidence:
        eid = item.get('evidence_id')
        if not isinstance(eid, str) or not eid or eid in ids:
            raise ValueError('Evidence IDs must be unique nonempty strings')
        ids.add(eid)
        for key in ('text', 'document_id', 'version', 'page', 'region'):
            if item.get(key) is None or (key == 'text' and not item[key].strip()):
                raise ValueError(f'Missing {key} on evidence {eid}')
        if type(item['page']) is not int or item['page'] < 1:
            raise ValueError('Page numbers must be one-based integers')
        if item.get('table'):
            table = item['table']
            if not table.get('columns') or not isinstance(table.get('rows'), list):
                raise ValueError('Table needs actual columns and rows')
            if any(len(row) != len(table['columns']) for row in table['rows']):
                raise ValueError('Canonical tables must be rectangular; preserve source spans in metadata')
    return ids


def validate_questions(questions, evidence, allow_unreviewed=False):
    ids = validate_evidence(evidence)
    if not questions:
        raise ValueError('Dataset is empty; no gold questions have been supplied')
    seen = set()
    by_id = {e['evidence_id']: e for e in evidence}
    for question in questions:
        qid = question.get('question_id')
        if not qid or qid in seen:
            raise ValueError('Question IDs must be unique and nonempty')
        seen.add(qid)
        if question.get('split') not in ('pilot', 'test', 'regression'):
            raise ValueError(f'Explicit pilot/test/regression split required: {qid}')
        if qid in ('Q1','Q2','Q3','Q4') and question['split'] != 'regression':
            raise ValueError('Q1–Q4 are regression only')
        for key in ('question', 'reference_answer', 'ground_truth_tuples', 'evidence_locations', 'categories'):
            if not question.get(key):
                raise ValueError(f'Missing {key}: {qid}')
        gold = question.get('gold_evidence_ids', [])
        required = question.get('required_evidence_ids', [])
        if not gold or not required or set(gold)-ids or set(required)-set(gold):
            raise ValueError(f'Invalid gold/required evidence: {qid}')
        for fact in question['ground_truth_tuples']:
            if not all(fact.get(k) for k in ('subject','attribute','evidence_ids')) or fact.get('value') is None:
                raise ValueError(f'Incomplete subject–attribute–value–condition tuple: {qid}')
            if 'condition' not in fact or set(fact['evidence_ids'])-set(gold):
                raise ValueError('Each fact needs explicit condition (possibly null) and gold source IDs')
        for eid in gold:
            source = by_id[eid]
            if not any(all(loc.get(k) == source.get(k) for k in ('document_id','version','page','region'))
                       for loc in question['evidence_locations']):
                raise ValueError(f'Gold location not traceable: {eid}')
        if not allow_unreviewed and (question.get('review_status') != 'approved' or not question.get('reviewer')):
            raise ValueError(f'Independent gold review required: {qid}')
