"""Persist expert attachments and a separate, tenant-scoped semantic graph."""
import uuid
from pathlib import Path
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from contracts.models import Contract, ContractVersion, ContractFile


def store_attachment(user, uploaded, encrypted_content, content_hash, saved_paths):
    name = Path(uploaded.name.replace('\\', '/')).name
    path = default_storage.save(f'contracts/expert/{uuid.uuid4().hex}/{name}', ContentFile(encrypted_content))
    saved_paths.append(path)
    contract = Contract.objects.create(company_id=user.company_id, contract_code='EXP-'+uuid.uuid4().hex[:20],
        title=name[:255], contract_type='Expert attachment', status='DRAFT')
    version = ContractVersion.objects.create(contract=contract, version_number=1,
        file_hash=content_hash, change_summary='Expert attachment uploaded by '+user.username)
    ContractFile.objects.create(version=version, file_name=name, file_path=default_storage.url(path),
                                file_size=uploaded.size, mime_type=uploaded.content_type or '')
    return version


def cleanup_uploads(paths):
    for path in paths:
        default_storage.delete(path)


def sync_semantic_graph(receipt, units, relations, document_scopes):
    from ai_extract.repositories.document_graph_repository import connection
    company = receipt.result['request']['company_id']
    root = f'a2-expert:{company}:{receipt.pk}'
    nodes = [dict(key=root+':'+u['id'], submission_key=root, company_id=company,
        kind=u['kind'], title=u['title'], quote=u['quote'], source_id=u['source_id'],
        source_url=u.get('source_url', ''), source_chunk_id=u.get('source_chunk_id', ''),
        author_id=receipt.user_id, review_status='draft') for u in units]
    edges = [dict(from_key=root if r['from_id']=='question' else root+':'+r['from_id'],
                  to_key=root if r['to_id']=='question' else root+':'+r['to_id'], kind=r['kind']) for r in relations]
    def write(tx):
        tx.run('MERGE (q:A2ExpertQuestion {key:$key}) SET q.company_id=$company,q.author_id=$author,'
               'q.question=$question,q.note=$note,q.review_status="draft"', key=root, company=company,
               author=receipt.user_id, question=receipt.query, note=receipt.result['request']['note']).consume()
        tx.run('MATCH (n:A2SemanticUnit {submission_key:$key}) DETACH DELETE n', key=root).consume()
        tx.run('MATCH (n:A2ExpertAnswer {submission_key:$key}) DETACH DELETE n', key=root).consume()
        if receipt.result['request'].get('answer'):
            tx.run('CREATE (a:A2ExpertAnswer {key:$answer_key,submission_key:$key,company_id:$company,'
                   'author_id:$author,text:$answer,review_status:"draft"})', answer_key=root+':answer',
                   key=root, company=company, author=receipt.user_id, answer=receipt.result['request']['answer']).consume()
        tx.run('MATCH (q:A2ExpertQuestion {key:$key})-[r:HAS_DOCUMENT]->() DELETE r', key=root).consume()
        tx.run('UNWIND $nodes AS row CREATE (n:A2SemanticUnit) SET n=row '
               'WITH n MATCH (q:A2ExpertQuestion {key:$key}) CREATE (q)-[:HAS_UNIT]->(n)', nodes=nodes, key=root).consume()
        for kind in ('ASKS_ABOUT', 'SUPPORTED_BY', 'QUALIFIES', 'CONTRADICTS', 'ANSWERS', 'HAS_UNIT'):
            tx.run('UNWIND $rows AS row MATCH (a {key:row.from_key}), (b {key:row.to_key}) '
                   'WHERE a.company_id=$company AND b.company_id=$company '
                   f'MERGE (a)-[:{kind} {{origin:"explicit_submission_binding",review_status:"draft"}}]->(b)',
                   rows=[r for r in edges if r['kind']==kind], company=company).consume()
        for unit in units:
            for concept in unit.get('concepts', []):
                # Hash includes kind and the literal normalized phrase. Company
                # is part of identity; shared vocabulary is not shared advice.
                concept_key=f'a2-concept:{company}:{concept["id"]}'
                tx.run('MERGE (c:A2Concept {key:$concept}) ON CREATE SET c.company_id=$company,'
                       'c.kind=$kind,c.value=$value '
                       'WITH c MATCH (u:A2SemanticUnit {key:$unit,company_id:$company}) '
                       f'MERGE (u)-[r:{concept["relation"]}]->(c) '
                       'SET r.quote=$quote,r.origin="source_grounded_term",r.review_status="draft"',
                       concept=concept_key,company=company,kind=concept['kind'],value=concept['value'],
                       unit=root+':'+unit['id'],quote=concept['quote']).consume()
        for scope in document_scopes:
            tx.run('MATCH (q:A2ExpertQuestion {key:$key}), (s:A3Snapshot {key:$snapshot,company_id:$company}) '
                   'MERGE (q)-[:HAS_DOCUMENT {origin:"expert_selected_or_uploaded_attachment"}]->(s)',
                   key=root, snapshot=scope['key'], company=company).consume()
            tx.run('MATCH (a:A2ExpertAnswer {key:$answer,company_id:$company}), '
                   '(s:A3Snapshot {key:$snapshot,company_id:$company}) '
                   'MERGE (a)-[:HAS_DOCUMENT {origin:"expert_attachment_not_verified_evidence"}]->(s)',
                   answer=root+':answer', snapshot=scope['key'],company=company).consume()
            for unit in units:
                if unit.get('snapshot_key') != scope['key']:
                    continue
                tx.run('MATCH (u:A2SemanticUnit {key:$unit,company_id:$company}), '
                       '(s:A3Snapshot {key:$snapshot,company_id:$company}) '
                       'MERGE (u)-[:HAS_EVIDENCE {origin:"exact_quote_in_document_source"}]->(s)',
                       unit=root+':'+unit['id'], company=company, snapshot=scope['key']).consume()
                tx.run('MATCH (u:A2SemanticUnit {key:$unit,company_id:$company}), '
                       '(b:A3Node {snapshot_key:$snapshot}) '
                       'WHERE b.key=$block OR (b.category="evidence" AND b.source_id=$source '
                       'AND b.text CONTAINS $quote) '
                       'MERGE (u)-[:HAS_EVIDENCE {origin:"exact_quote_in_extracted_block"}]->(b)',
                       unit=root+':'+unit['id'], company=company, snapshot=scope['key'], block=unit['graph_block_key'],
                       source=unit.get('source_chunk_id',''),quote=unit['quote']).consume()
    with connection() as session:
        for label in ('A2ExpertQuestion', 'A2ExpertAnswer', 'A2SemanticUnit', 'A2Concept'):
            session.run(f'CREATE CONSTRAINT {label.lower()}_key IF NOT EXISTS '
                        f'FOR (n:{label}) REQUIRE n.key IS UNIQUE').consume()
        session.execute_write(write)
        counts = session.run('MATCH (q:A2ExpertQuestion {key:$key}) '
            'RETURN COUNT { (q)-[:HAS_UNIT]->() } AS units, '
            'COUNT { (q)-[:HAS_DOCUMENT]->() } AS documents', key=root).single()
        evidence = session.run('MATCH (u:A2SemanticUnit {submission_key:$key})-[r:HAS_EVIDENCE]->() '
                               'RETURN count(r) AS links', key=root).single()['links']
        missing = session.run('UNWIND $keys AS key MATCH (u:A2SemanticUnit {key:key}) '
            'WHERE NOT (u)-[:HAS_EVIDENCE]->(:A3Node) RETURN count(u) AS missing',
            keys=[root+':'+u['id'] for u in units if u.get('snapshot_key')]).single()['missing']
    complete = counts['units'] == len(nodes) and counts['documents'] == len(document_scopes) and missing == 0
    return dict(status='ready' if complete else 'partial', key=root, units=counts['units'],
                proposed_relations=len(edges), documents=counts['documents'], evidence_links=evidence,
                missing_source_units=missing, review_status='draft')


def related_units(receipt, limit=12):
    """Same-company shared-term navigation, including explicitly marked drafts."""
    from ai_extract.repositories.document_graph_repository import connection
    company=receipt.result['request']['company_id']
    root=f'a2-expert:{company}:{receipt.pk}'
    query='''
    MATCH (seed:A2SemanticUnit {submission_key:$root,company_id:$company})
          -[:ABOUT_MACHINE|HAS_ACTION|HAS_SOLUTION]->(c:A2Concept {company_id:$company})
          <-[:ABOUT_MACHINE|HAS_ACTION|HAS_SOLUTION]-(other:A2SemanticUnit {company_id:$company})
    WHERE other.submission_key<>$root
    WITH other,collect(DISTINCT {kind:c.kind,value:c.value}) AS common
    RETURN other.key AS key,other.submission_key AS submission_key,other.title AS title,
           other.quote AS quote,other.source_url AS source_url,other.review_status AS review_status,common
    ORDER BY size(common) DESC,key LIMIT $limit
    '''
    with connection() as session:
        return [dict(row) for row in session.run(query,root=root,company=company,limit=min(30,max(1,limit)))]


def visible_submissions(user):
    from contracts.models import KnowledgeSearchLog
    rows=KnowledgeSearchLog.objects.filter(scope='expert',status='COMPLETED').select_related('user')
    if not user.is_authenticated:return rows.none()
    if user.is_superuser:return rows
    return rows.filter(result__request__company_id=user.company_id) if user.company_id else rows.none()


def submissions_by_ids(identities):
    from contracts.models import KnowledgeSearchLog
    return KnowledgeSearchLog.objects.filter(pk__in=identities,status='COMPLETED',scope='expert')


def get_visible_submission(user, identity):
    return visible_submissions(user).get(pk=identity)
