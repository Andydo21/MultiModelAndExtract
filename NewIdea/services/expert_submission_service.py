"""Expert question -> A3 attachments -> Qwen semantic units -> scoped Neo4j."""
import hashlib
from pathlib import Path
from django.db import transaction
from django.conf import settings
from NewIdea.repositories import contract_repository, knowledge_repository, expert_submission_repository
from .search_job_service import reserve_search
from .errors import KnowledgeError
from .search_evidence_service import submission_records

KINDS = {'clause', 'issue', 'condition', 'recommendation', 'fact'}
RELATIONS = {'ASKS_ABOUT', 'SUPPORTED_BY', 'QUALIFIES', 'CONTRADICTS', 'ANSWERS'}
CONCEPT_RELATIONS = {'machine': 'ABOUT_MACHINE', 'action': 'HAS_ACTION', 'solution': 'HAS_SOLUTION'}


def grounded_concepts(rows, quote):
    """Share literal terms, never model-generated synonyms or unquoted machine IDs."""
    if not isinstance(rows, list) or len(rows) > 20:
        raise ValueError('Invalid concepts')
    concepts = {}
    for row in rows:
        if isinstance(row, dict):
            # Identity is copied from the verified source span, not a model
            # translation/canonical name. Old explicit values remain validated.
            row=dict(row,value=row.get('value',row.get('quote')))
        if (not isinstance(row, dict) or row.get('kind') not in CONCEPT_RELATIONS
            or not isinstance(row.get('quote'), str) or not row['quote'].strip()
            or row['quote'] not in quote or not isinstance(row.get('value'), str)
            or not row['value'].strip() or row['value'] not in row['quote'] or len(row['value']) > 400):
            raise ValueError('Ungrounded concept')
        normalized = ' '.join(row['value'].split()).casefold()
        key = hashlib.sha256((row['kind']+'\0'+normalized).encode()).hexdigest()
        concepts[key] = dict(id=key, kind=row['kind'], value=row['value'], quote=row['quote'],
                             relation=CONCEPT_RELATIONS[row['kind']])
    return list(concepts.values())


def enqueue(user, data, files):
    if not user.company_id:
        raise KnowledgeError('Tài khoản cần thuộc công ty.', 403)
    question, note = data.get('question', ''), data.get('note', '')
    answer = data.get('answer', '')
    if not isinstance(question, str) or not 1 <= len(question.strip()) <= 2000:
        raise ValueError('Nhập câu hỏi từ 1 đến 2000 ký tự.')
    if not isinstance(note, str) or len(note) > 12000:
        raise ValueError('Ghi chú tối đa 12000 ký tự.')
    if not isinstance(answer, str) or len(answer) > 12000:
        raise ValueError('Câu trả lời tối đa 12000 ký tự.')
    if len(files) > 3 or sum(f.size for f in files) > 30*1024*1024:
        raise ValueError('Tối đa 3 file, tổng dung lượng 30 MB.')
    for f in files:
        if f.size > 15*1024*1024 or f.size == 0 or Path(f.name).suffix.lower() not in {'.pdf','.png','.jpg','.jpeg','.docx','.txt'}:
            raise ValueError('File PDF, PNG, JPG, DOCX hoặc TXT, tối đa 15 MB mỗi file.')
    selected = data.get('version_id')
    version = contract_repository.get_visible_version(user, int(selected)) if selected else None
    if version and version.contract.company_id != user.company_id:
        raise KnowledgeError('Tài liệu phải thuộc cùng công ty với yêu cầu chuyên gia.', 403)
    saved = []
    try:
        with transaction.atomic():
            receipt = reserve_search(user, question, version, generated=True, scope='expert')
            if receipt is None:
                raise KnowledgeError('Đã đạt giới hạn xử lý; thử lại sau một phút.', 429)
            from contracts.services.document_encryption_service import encrypt_pdf
            attachments = []
            for file in files:
                content=file.read()
                attachments.append(expert_submission_repository.store_attachment(
                    user,file,encrypt_pdf(content),hashlib.sha256(content).hexdigest(),saved).pk)
            receipt.status = 'QUEUED'
            receipt.result = dict(request=dict(scope='expert', company_id=user.company_id,
                question=question.strip(), note=note.strip(), answer=answer.strip(), version_id=version.pk if version else None,
                attachment_version_ids=attachments))
            knowledge_repository.save_search(receipt, 'status', 'result')
        return dict(success=True, search_id=receipt.pk, status='QUEUED', attachment_version_ids=attachments)
    except Exception:
        expert_submission_repository.cleanup_uploads(saved)
        raise


def validate_semantics(result, sources, prefix):
    rows, proposals = result.get('units'), result.get('relations', [])
    if not isinstance(rows, list) or len(rows) > 40 or not isinstance(proposals, list) or len(proposals) > 80:
        raise ValueError('Invalid semantic extraction schema')
    source_map = {s['id']:s for s in sources}
    units, mapping, identities = [], {'question':'question'}, set()
    if any(s['id'].startswith('expert-answer:') for s in sources):
        mapping['answer']='answer'
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get('id'), str) or not row['id']
            or row['id'] in mapping or row.get('kind') not in KINDS
            or row.get('source_id') not in source_map or not isinstance(row.get('quote'), str)
            or not row['quote'].strip() or row['quote'] not in source_map[row['source_id']]['text']
            or not isinstance(row.get('title'), str) or not 1 <= len(row['title']) <= 200):
            raise ValueError('Ungrounded semantic unit')
        identity = prefix+':'+hashlib.sha256((row['source_id']+'\0'+row['quote']+'\0'+row['kind']).encode()).hexdigest()[:24]
        if identity in identities:
            raise ValueError('Duplicate semantic unit')
        identities.add(identity)
        mapping[row['id']] = identity
        source = source_map[row['source_id']]
        units.append(dict(id=identity, kind=row['kind'], title=row['title'], quote=row['quote'],
                          concepts=grounded_concepts(row.get('concepts', []), row['quote']),
                          source_id=row['source_id'], source_url=source.get('source_url', ''),
                          source_chunk_id=source.get('source_chunk_id', ''),
                          snapshot_key=source.get('snapshot_key', ''), graph_block_key=source.get('graph_block_key', '')))
    links = []
    for row in proposals:
        if (not isinstance(row, dict) or row.get('from_id') not in mapping or row.get('to_id') not in mapping
            or row.get('kind') not in RELATIONS or row['from_id']==row['to_id']):
            raise ValueError('Invalid semantic relation')
        links.append(dict(from_id=mapping[row['from_id']], to_id=mapping[row['to_id']], kind=row['kind']))
    return units, links


def process(receipt):
    from ai_extract.services.contract_pipeline_service import LayoutExtractionService, request_qwen
    from ai_extract.services.document_graph_service import graph_scope
    data = receipt.result['request']
    if receipt.user.company_id != data['company_id']:
        raise KnowledgeError('Company scope changed', 403)
    versions = list(dict.fromkeys(data['attachment_version_ids']+([data['version_id']] if data.get('version_id') else [])))
    sources = [dict(id='question', text=receipt.query)]
    if data['note']:
        sources.append(dict(id='expert-note', text=data['note']))
    if data.get('answer'):
        sources.append(dict(id='expert-answer', text=data['answer']))
    scopes, documents = [], []
    for identity in versions:
        version = contract_repository.get_visible_version(receipt.user, identity)
        run = contract_repository.completed_extraction_run(version)
        # Uploaded attachments are new immutable versions, using the same A3.
        if not run and identity in data['attachment_version_ids']:
            receipt.result['progress'] = dict(stage='a3_extraction', version_id=identity)
            knowledge_repository.save_search(receipt, 'result')
            run = LayoutExtractionService().extract_version(version)
        if not run:
            raise ValueError('Selected document has no completed A3 extraction')
        scope = graph_scope(run, run.structure)
        scopes.append(scope)
        documents.append(dict(version_id=version.pk, contract_id=version.contract_id,
            title=version.contract.title, extraction_status=run.status,
            dense_status=run.dense_index_status, colpali_status=run.colpali_index_status,
            source_url=f'/ai-extract/contracts/{version.contract_id}/?version_id={version.pk}'))
        for r in contract_repository.version_sources(version):
            if r['text'].strip():
                sources.append(dict(id=f'v{identity}:{r["chunk_id"]}', text=r['text'], source_url=r['source_url'],
                    source_chunk_id=r['chunk_id'], snapshot_key=scope['key'],
                    graph_block_key=scope['key']+f':layout-block:{r["page_number"]}:{r["chunk_id"]}'))
    batches = []
    for source in sources:
        # Split long source text without generating or rewriting it.
        for offset in range(0, len(source['text']), 2500):
            unit = dict(source, id=source['id']+f':{offset}', text=source['text'][offset:offset+2500])
            # Prevent facts from an answer/document being attributed to the
            # question or another block by a small generation model.
            batches.append([unit])
    units, relations, errors, qwen_outputs = [], [], [], []
    for index, batch in enumerate(batches[:12]):
        receipt.result['progress'] = dict(stage='qwen_semantic_extraction', batch=index+1, batches=min(12,len(batches)))
        knowledge_repository.save_search(receipt, 'result')
        try:
            for attempt in range(2):
                payload=dict(question=receipt.query,sources=batch)
                if attempt:
                    payload['validation_feedback']='Previous extraction failed source/schema validation. Copy quotes and concept values exactly from this source only; omit uncertain concepts.'
                result = request_qwen('expert_semantic_extract', payload)
                qwen_outputs.extend(dict(item, stage='semantic_extraction', batch=index+1, attempt=attempt+1)
                                    for item in result.get('qwen_outputs', []))
                try:
                    extracted, proposed = validate_semantics(result, batch, str(index))
                    break
                except ValueError:
                    if attempt:raise
            units.extend(extracted)
            relations.extend(proposed)
        except (RuntimeError, ValueError, TypeError, KeyError) as error:
            qwen_outputs.extend(dict(item, stage='failed', batch=index+1)
                                for item in getattr(error, 'qwen_outputs', []))
            errors.append(dict(batch=index+1, error_type=type(error).__name__))
            if len(errors) >= 2: break
    if data.get('answer'):
        relations.append(dict(from_id='answer', to_id='question', kind='ANSWERS'))
    for unit in units:
        from_answer = unit['source_id'].startswith('expert-answer:')
        relations.append(dict(from_id='answer' if from_answer else 'question', to_id=unit['id'],
                              kind='HAS_UNIT'))
    relations=list({(r['from_id'],r['to_id'],r['kind']):r for r in relations}.values())
    receipt.result['semantic_analysis'] = dict(units=units, relations=relations, review_status='draft')
    receipt.result['qwen_outputs'] = qwen_outputs
    knowledge_repository.save_search(receipt, 'result')
    try:
        graph = expert_submission_repository.sync_semantic_graph(receipt, units, relations, scopes)
    except Exception as error:
        graph = dict(status='unavailable', error_type=type(error).__name__)
    try:
        related = expert_submission_repository.related_units(receipt, limit=12)
    except Exception:
        related = []
    # Text knowledge uses A3's BGE indexer directly, never OCR or ColPali.
    from ai_extract.services.contract_pipeline_service import PlanACollectionIndexer, IndexResult
    receipt.result['progress'] = dict(stage='indexing_knowledge')
    knowledge_repository.save_search(receipt, 'result')
    indexed = (IndexResult.capture(lambda: PlanACollectionIndexer().index_text_records(
        submission_records(receipt))) if getattr(settings, 'A3_HYBRID_ENABLED', True)
        else IndexResult('SKIPPED', error='Neural indexing disabled'))
    receipt.result['semantic_index'] = dict(status=indexed.status, points=indexed.indexed_count, error=indexed.error)
    knowledge_repository.save_search(receipt, 'result')
    return dict(success=bool(units), submission_id=receipt.pk, question=receipt.query, answer=data.get('answer',''), units=units,
        qwen_outputs=qwen_outputs,
        relations=relations, documents=documents, graph=graph, vector_index=receipt.result['semantic_index'], related_units=related, requires_expert_review=True,
        analysis_status='unavailable' if errors and not units else 'partial' if errors or len(batches)>12 else 'completed',
        coverage=dict(batches_total=len(batches), batches_attempted=index+1 if batches else 0,
                      max_batches=12, errors=errors, semantic_correctness_verified=False))


def read_submission(user, identity):
    if not str(identity).isdecimal():
        raise ValueError('Invalid submission ID')
    receipt=expert_submission_repository.get_visible_submission(user,int(identity))
    if not submission_records(receipt):
        raise KnowledgeError('Submission not found.',404)
    data = receipt.result['request']
    return dict(success=True, submission_id=receipt.pk, author=receipt.user.username, review_status='draft',
        question=receipt.query, answer=data.get('answer', ''), note=data.get('note', ''),
        units=receipt.result.get('semantic_analysis', {}).get('units', []),
        relations=receipt.result.get('semantic_analysis', {}).get('relations', []),
        qwen_outputs=receipt.result.get('qwen_outputs', []),
        vector_index=receipt.result.get('semantic_index', {}))

