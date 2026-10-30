"""JSON/JSONL input and document-disjoint validation; no sample data generation."""
import json
from pathlib import Path

from .schema import validate_questions


def load_records(path):
    path = Path(path)
    if path.suffix.lower() == '.jsonl':
        with path.open(encoding='utf-8-sig') as stream:
            return [json.loads(line) for line in stream if line.strip()]
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(data, list):
        raise ValueError('Expected a JSON array or JSONL records')
    return data


def load_dataset(path, evidence, allow_unreviewed=False):
    records = load_records(path)
    validate_questions(records, evidence, allow_unreviewed)
    documents = {}
    for question in records:
        for location in question['evidence_locations']:
            document = location['document_id']
            previous = documents.setdefault(document, question['split'])
            if previous != question['split']:
                raise ValueError(f'Document crosses dataset splits: {document}')
    # Document-family/near-duplicate leakage requires an external reviewed split manifest.
    return records
