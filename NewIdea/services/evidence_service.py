"""Bind approved case evidence to a specific persisted contract version."""
import hashlib
from NewIdea.repositories import contract_repository

def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def bind_evidence(case):
    record = next((r for r in contract_repository.version_sources(case.version) if r['chunk_id'] == case.source_reference), None)
    if not record:
        raise ValueError('Chọn ID nguồn A3 còn tồn tại trong đúng phiên bản (block:…, clause:… hoặc context:…).')
    return dict(chunk_id=record['chunk_id'], version_id=case.version_id,
                text=record['text'], sha256=digest(record['text']),
                source_url=record['source_url'], source_location=record['source_location'])


def evidence_current(case):
    if not case.evidence_snapshot:
        return False
    try:
        return bind_evidence(case)['sha256'] == case.evidence_snapshot.get('sha256')
    except ValueError:
        return False


