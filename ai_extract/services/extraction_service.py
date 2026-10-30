"""Coordinate persisted summaries, entity extraction and A3 clause extraction."""
from ai_extract.repositories import ContractSummaryRepository, ExtractedEntityRepository
from ai_extract.repositories import extraction_repository

import logging
from contracts.models import ContractVersion
from contracts.repositories import ClauseRepository
from ai_extract.models import Clause
from ai_extract.repositories import clause_extract_data_repository as data_repository
from .contract_pipeline_service import request_qwen

logger = logging.getLogger('ai_extract')
MODEL_ID = 'Qwen/Qwen2.5-VL-7B-Instruct-AWQ'


class SummarizeService:
    """Generate and persist a version's executive summary."""

    def summarize_version(self, version: ContractVersion) -> dict:
        """Call the Qwen gateway and persist its summary."""
        clauses = extraction_repository.clauses_for_version(version)
        if not clauses:
            raise ValueError(f'ContractVersion {version.id} has no clauses. Run clause extraction first.')
        payload = {'clauses': [{'title': cl.clause_title, 'content': cl.clause_content} for cl in clauses], 'contract_metadata': {'contract_code': version.contract.contract_code, 'title': version.contract.title, 'version': version.version_number}}
        logger.info(f'[summarize] Qwen chat completion – version_id={version.id}, {len(clauses)} clauses')
        try:
            data = request_qwen('summarize', payload)
        except RuntimeError as e:
            raise RuntimeError(f'Qwen summarize error: {e}') from e
        summary_text = data.get('summary', '')
        obj = ContractSummaryRepository.create_or_update(version=version, summary=summary_text, model_id=MODEL_ID)
        logger.info(f'[summarize] Saved ContractSummary id={obj.id}')
        return {'version_id': version.id, 'contract_id': version.contract.id, 'summary': summary_text, 'model_id': MODEL_ID}



class ExtractEntityService:
    """Persist Qwen clause fields for the existing contract extraction API."""

    def extract_version(self, version: ContractVersion, re_extract: bool=False) -> list:
        """Run entity extraction for every clause in a ContractVersion."""
        if re_extract:
            deleted = ExtractedEntityRepository.delete_kaggle_entities_for_version(version)
            logger.info(f'[extract] Deleted {deleted} existing Kaggle entities for version {version.id}')
        clauses = extraction_repository.clauses_for_version(version)
        if not clauses:
            logger.info(f'[extract] No clauses found for version {version.id}. Running ClauseExtractService first.')
            clause_extractor = ClauseExtractService()
            clause_extractor.extract_version(version, re_extract=True)
            clauses = extraction_repository.clauses_for_version(version)
            if not clauses:
                raise ValueError(f'ContractVersion {version.id} has no clauses after splitting.')
        results = []
        for clause in clauses:
            text = f'[{clause.clause_title}]\n{clause.clause_content}'
            logger.info(f"[extract] Qwen chat completion – clause_id={clause.id} '{clause.clause_title[:40]}'")
            try:
                data = request_qwen('extract_entities', {'text': text})
            except RuntimeError as e:
                logger.error(f'[extract] HTTP error for clause {clause.id}: {e}')
                raise RuntimeError(f'Không thể kết nối đến máy chủ AI (AI Service communication failed): {e}')
            raw_entities = data.get('entities', {})
            saved = ExtractedEntityRepository.save_from_kaggle(clause=clause, raw_entities=raw_entities)
            logger.info(f"[extract] Saved {len(saved)} ExtractedEntity rows for clause '{clause.clause_title[:40]}'")
            results.append({'clause_id': clause.id, 'clause_title': clause.clause_title, 'entities': raw_entities, 'saved_count': len(saved), 'saved_ids': [e.id for e in saved]})
        return results


    def extract_from_text(self, text: str, clause: Clause=None) -> dict:
        """Extract entities from arbitrary raw text."""
        logger.info(f'[extract] Qwen chat completion – free-text len={len(text)}')
        try:
            data = request_qwen('extract_entities', {'text': text})
        except RuntimeError as e:
            raise RuntimeError(f'Qwen extract_entities error: {e}') from e
        raw_entities = data.get('entities', {})
        saved_count = 0
        if clause:
            saved = ExtractedEntityRepository.save_from_kaggle(clause=clause, raw_entities=raw_entities)
            saved_count = len(saved)
            logger.info(f'[extract] Saved {saved_count} free-text ExtractedEntity rows')
        return {'entities': raw_entities, 'saved_count': saved_count}


class ClauseExtractService:
    """
    Materialize the existing A3 article hierarchy into the web's Clause rows.
    Extraction and retrieval share the same legal boundaries and source text.
    """

    def extract_version(self, version: ContractVersion, re_extract: bool=False) -> dict:
        """Create clauses from the A3 source hierarchy, without another splitter."""
        from django.db import transaction
        from ai_extract.services.contract_pipeline_service import clauses_from_structure
        from ai_extract.services.contract_pipeline_service import LayoutExtractionService
        run = data_repository.first_layout_extraction_runs_for_extract_version(version)
        if run is None and (not data_repository.exists_contexts_for_extract_version(version)):
            if data_repository.exists_files_for_extract_version(version):
                run = LayoutExtractionService().extract_version(version)
            elif not data_repository.exists_ai_extract_clauses_for_extract_version(version):
                raise ValueError('No extracted source or file for this version.')
        if run:
            from ai_extract.repositories import layout_repository
            records = layout_repository.structure_records(run)
        else:
            records = [dict(chunk_id=f'context:{c.pk}', text=c.content, page_number=index) for index, c in enumerate(data_repository.order_by_contexts_for_extract_version(version), 1)]
        from ai_extract.services.document_structure_service import structure_for_version
        tree = structure_for_version(version, records, version.contract.title)
        candidates = clauses_from_structure(tree)
        if not candidates and (not data_repository.exists_ai_extract_clauses_for_extract_version(version)):
            raise ValueError('A3 hierarchy contains no source text.')
        with transaction.atomic():
            extraction_repository.lock_version(version)
            if candidates and (re_extract or not data_repository.exists_ai_extract_clauses_for_extract_version(version)):
                extraction_repository.delete_version_clauses(version)
                repo = ClauseRepository()
                for item in candidates:
                    repo.create_clause(version, item['title'], item['content'])
            saved = list(data_repository.all_ai_extract_clauses_for_extract_version(version))
        return dict(version_id=version.pk, contract_id=version.contract_id, total_clauses=len(saved), clauses=[dict(clause_id=c.pk, clause_title=c.clause_title, clause_type=c.clause_type, content_length=len(c.clause_content)) for c in saved])



class ExtractionError(ValueError):

    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status

def select_version(user, contract_id, version_id=None):
    contract = extraction_repository.contract_for(user, contract_id)
    version = extraction_repository.version_for(contract, version_id)
    if version is None:
        raise ExtractionError('Contract has no version yet.')
    return (contract, version)

def summarize(user, contract_id, data):
    _, version = select_version(user, contract_id, data.get('version_id'))
    return SummarizeService().summarize_version(version)

def summary(user, contract_id):
    contract, version = select_version(user, contract_id)
    obj = ContractSummaryRepository.get_by_version(version)
    if not obj:
        raise ExtractionError('No summary found. Run summarize first.', 404)
    return dict(id=obj.id, contract_id=contract.id, version_id=version.id, summary=obj.summary, model_id=obj.model_id, created_at=obj.created_at.isoformat(), updated_at=obj.updated_at.isoformat())

def extract_entities(user, contract_id, data):
    contract, version = select_version(user, contract_id, data.get('version_id'))
    results = ExtractEntityService().extract_version(version, re_extract=bool(data.get('re_extract', False)))
    return dict(contract_id=contract.id, version_id=version.id, total_clauses=len(results), results=results)

def extract_text(user, data):
    text = data.get('text', '')
    if not isinstance(text, str) or not text.strip():
        raise ExtractionError("Field 'text' is required.")
    clause = extraction_repository.clause_for(user, data['clause_id']) if data.get('clause_id') else None
    return ExtractEntityService().extract_from_text(text.strip(), clause=clause)

def entities(user, contract_id, version_id=None):
    contract, version = select_version(user, contract_id, version_id)
    records = ExtractedEntityRepository.get_by_version(version)
    items = [dict(id=e.id, clause_id=e.clause_id, clause_title=e.clause.clause_title, entity_type=e.entity_type, entity_value=e.entity_value, normalized_value=e.normalized_value, confidence_score=float(e.confidence_score)) for e in records]
    return dict(contract_id=contract.id, version_id=version.id, count=len(items), entities=items)

def extract_clauses(user, contract_id, data):
    _, version = select_version(user, contract_id, data.get('version_id'))
    return ClauseExtractService().extract_version(version, re_extract=bool(data.get('re_extract', False)))

def clauses(user, contract_id, version_id=None):
    contract, version = select_version(user, contract_id, version_id)
    items = [dict(id=c.id, clause_title=c.clause_title, clause_type=c.clause_type, content_length=len(c.clause_content), clause_content=c.clause_content) for c in extraction_repository.clauses_for_version(version)]
    return dict(contract_id=contract.id, version_id=version.id, count=len(items), clauses=items)
