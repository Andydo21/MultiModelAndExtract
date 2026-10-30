"""Build source-bound A2 expert citations and discover indexed structure nodes."""
from urllib.parse import urlencode
import math
import hashlib
import re
from NewIdea.repositories import contract_repository, expert_submission_repository
from django.conf import settings
from ai_extract.services.document_graph_service import graph_scope
from shared.retrieval import evidence_identity, tokens
from .case_rules import CASE_FIELDS, serialize_case
from .evidence_service import evidence_current


def find_expert_cases(query, cases, limit=3):
    """Rank actual, current approved cases with the multilingual neural reranker."""
    from ai_extract.services.contract_pipeline_service import HybridBackend
    records = [
        dict(case=case, text=' '.join(getattr(case, field) for field in CASE_FIELDS),
             original_name=case.title)
        for case in cases if case.status == 'approved' and not case.title.startswith('[DEMO]')
        and evidence_current(case)
    ]
    if not records:
        return []
    ranked = HybridBackend().rerank_many([(query, records)])[0]
    return [
        dict(serialize_case(hit['case']), match_score=hit['neural_logit'],
             requires_applicability_review=True, score_kind='neural_reranker_not_applicability')
        for hit in ranked[:limit]
    ]


def select_retrieval_seeds(candidates, limit=None):
    """Cover requested parts, then balance ranked relevance and marginal information.

    Ranks are produced by the existing retrieval/reranker, never inferred from
    document language or domain. Similar passages are downranked, not discarded:
    different versions, quantities, conditions and provenance remain available.
    """
    limit = limit if limit is not None else getattr(settings, 'A2_ANSWER_SOURCE_LIMIT', 24)
    # Keep legacy callers' order when they have no cross-query ranking metadata.
    if not any(c.get('retrieval_ranks') for c in candidates):
        return candidates[:limit]
    remaining = list(range(len(candidates)))
    selected, covered, documents = [], set(), set()
    words, parts, relevance, fusion, docs = [], [], [], [], []
    for index, source in enumerate(candidates):
        words.append(set(tokens(source.get('text', ''))))
        parts.append({h['part'] for h in source.get('retrieval_hints', [])})
        ranks = [h['rank'] for h in source.get('retrieval_ranks', [])] or [index+1]
        relevance.append(max(1 / (1 + math.log2(rank)) for rank in ranks))
        fusion.append(sum(1 / (60 + rank) for rank in ranks))
        docs.append(source.get('document_id') or source.get('version_id'))
    overlap = [0.] * len(candidates)
    while remaining and len(selected) < limit:
        # Coverage first; novelty adjusts relevance without equal document quotas.
        chosen = max(remaining, key=lambda i: (len(parts[i] - covered),
            relevance[i] - 0.35 * overlap[i] + (0.05 if docs[i] not in documents else 0), fusion[i]))
        remaining.remove(chosen)
        selected.append(candidates[chosen])
        covered.update(parts[chosen]); documents.add(docs[chosen])
        for i in remaining:
            overlap[i] = max(overlap[i], len(words[i] & words[chosen]) / max(1, len(words[i] | words[chosen])))
    return selected


def visual_source_context(seeds, leaves):
    """Keep Surya child text and nearby caption candidates with selected figures."""
    from ai_extract.services.document_graph_service import figure_caption_candidates
    pages = {}
    for leaf in leaves:
        pages.setdefault(leaf.get('page_number'), []).append(leaf)
    output = []
    for figure in seeds:
        if figure.get('layout_type') != 'figure' or not figure.get('chunk_id'):
            continue
        page = pages.get(figure.get('page_number'), [])
        meta = figure.get('structure_metadata') or {}
        region = meta.get('surya_region_id')
        target = figure['chunk_id']
        for leaf in page:
            child = leaf.get('structure_metadata') or {}
            if (region and child.get('surya_parent_region_id') == region
                    and child.get('source_parent_status') != 'unresolved'
                    and leaf.get('text','').strip() and not leaf.get('retrieval_excluded')):
                output.append(dict(leaf, expansion_reason='visual_source_child',
                    visual_context_links=[dict(figure_chunk_id=target, kind='surya_child')]))
        for candidate_figure,caption,edge in figure_caption_candidates(page):
            if candidate_figure['chunk_id']==target:
                kind='nearby_caption_candidate' if caption.get('layout_type')=='figure_caption' else 'nearby_text_candidate'
                output.append(dict(caption, expansion_reason='visual_caption_candidate',
                    visual_context_links=[dict(figure_chunk_id=target, kind=kind,
                                               spatial_relation=edge['kind'])]))
    return output


def select_answer_sources(candidates, limit=None):
    """Bound ranked seeds, preserving all source context of admitted seeds."""
    limit = limit if limit is not None else getattr(settings, 'A2_ANSWER_SOURCE_LIMIT', 24)
    matches, context, seen, text_seen = [], [], {}, {}
    for contextual in (False, True):
        for source in candidates:
            identity = evidence_identity(source)
            version = source.get('document_id') or source.get('version_id')
            text_key = (version, source.get('source_role'), source.get('evidence_kind'),
                        tuple(source.get('heading_path',[])),
                        ' '.join(source.get('text', '').split()) or identity)
            is_context = source.get('expansion_reason', 'query_match') != 'query_match'
            if is_context != contextual:
                continue
            previous = seen.get(identity) or text_seen.get(text_key)
            if previous is not None:
                for key in ('retrieval_hints', 'retrieval_ranks', 'visual_context_links'):
                    if source.get(key):
                        hints = previous.setdefault(key, [])
                        hints.extend(h for h in source[key] if h not in hints)
                continue
            source = dict(source)
            for key in ('retrieval_hints', 'retrieval_ranks'):
                if key in source:
                    source[key] = list(source[key])
            (context if contextual else matches).append(source)
            seen[identity] = text_seen[text_key] = source
    selected = matches[:limit]
    represented = {s.get('document_id') or s.get('version_id') for s in selected}
    selected.extend(s for s in context
                    if (s.get('document_id') or s.get('version_id')) in represented)
    return [dict(source, evidence_id=f'E{index}') for index, source in enumerate(selected, 1)]


EXPERT_INSTRUCTION = ('Distinguish document facts from approved expert opinions. '
    'Expert cases are conditional advice, not facts established by the source document. State their author, '
    'revision and applicability; do not assume they apply to the current document. '
    'Never omit an expert case applicability condition or its verification limits. '
    'Expert units marked draft are unreviewed user submissions, not approved procedures. '
    'Expert questions describe what someone asked, never evidence that its premise is true. '
    'State the author and review status when citing expert submissions; preserve conditions.')


def with_expert_evidence(citations, cases):
    evidence = [dict(c, evidence_kind=c.get('evidence_kind', 'document')) for c in citations]
    for case in cases[:3]:
        snapshot = case.get('evidence_snapshot') or {}
        if case.get('is_demo') or case.get('status') != 'approved' or not snapshot.get('sha256') or not snapshot.get('text'):
            continue
        identifier = f'E{len(evidence)+1}'
        # Full conditions remain intact, rather than keyword-windowing advice.
        text = '\n'.join(f'{label}: {case.get(field, "")}' for label, field in (
            ('Issue', 'issue'), ('Reasoning', 'reasoning'), ('Resolution', 'resolution'),
            ('Verification', 'verification'), ('Applicability', 'applicability')))
        text += '\nSupporting source: '+snapshot['text']
        evidence.append(dict(evidence_id=identifier, evidence_kind='expert_opinion',
            is_demo=case.get('is_demo', False),
            chunk_id=case['evidence_id'], expert_case_id=case['case_id'], revision=case['revision'],
            author=case['author'], reviewer=case.get('reviewer'), applicability=case.get('applicability', ''),
            requires_applicability_review=True, text=text, original_name='Ý kiến chuyên gia: '+case['title'],
            contract_id=case['contract_id'], version_id=case['version_id'],
            source_location=f"{case['author']} · revision {case['revision']} · "+snapshot.get('source_location', ''),
            source_url=snapshot.get('source_url', ''), heading_path=[],
            source_chunk_id=snapshot['chunk_id']))
    return evidence


def matched_structure_nodes(tree, citations, version, run=None):
    """Node lookup through authorized retrieved leaves, not unscoped graph names."""
    by_id = {n['id']: n for n in tree.get('nodes', [])}
    matches = {}
    scope = graph_scope(run, tree) if run and tree.get('graph_index', {}).get('status') == 'ready' else None
    if scope and tree['graph_index'].get('fingerprint') != scope['fingerprint']:
        scope = None
    for citation in citations:
        if citation.get('evidence_kind') == 'expert_opinion':
            continue
        source_identity = 'leaf:'+citation.get('chunk_id', '')
        source_node = dict(node_id=source_identity, title=citation.get('text', '')[:100] or 'Vùng ảnh',
            kind=citation.get('layout_type', 'evidence'), contract_id=version.contract_id, version_id=version.pk,
            evidence_ids=[citation.get('evidence_id')], retrieval_basis='retrieved_source_leaf',
            vector_unit='source_leaf', vector_index_status=getattr(run, 'dense_index_status', 'NOT_INDEXED'),
            source_url=citation.get('source_url', ''))
        if scope:
            params = urlencode(dict(version_id=version.pk, graph_node=scope['key']+':'+source_identity))
            source_node['graph_url'] = f'/ai-extract/contracts/{version.contract_id}/pages/{citation.get("page_number", 1)}/?{params}#neo4j-graph'
        matches.setdefault(source_identity, source_node)
        for identity in citation.get('ancestor_ids', []):
            node = by_id.get(identity)
            if not node or node.get('kind') == 'document':
                continue
            item = matches.setdefault(identity, dict(node_id=identity, title=node['title'], kind=node['kind'],
                contract_id=version.contract_id, version_id=version.pk, evidence_ids=[],
                retrieval_basis='retrieved_leaf_ancestor', independently_vectorized=False,
                source_url=citation.get('source_url', '')))
            if citation.get('evidence_id') not in item['evidence_ids']:
                item['evidence_ids'].append(citation.get('evidence_id'))
            if scope:
                page = citation.get('page_number', 1)
                params = urlencode(dict(version_id=version.pk, graph_node=scope['key']+':node:'+identity))
                item['graph_url'] = f'/ai-extract/contracts/{version.contract_id}/pages/{page}/?{params}#neo4j-graph'
    return list(matches.values())[:30]


def text_record(record):
    """Stable source identity plus a fingerprint checked before using a hit."""
    import json
    record = dict(record, source_kind='knowledge_text')
    record['content_sha256'] = hashlib.sha256(json.dumps(record, sort_keys=True,
        ensure_ascii=False, default=str).encode()).hexdigest()
    return record


def submission_records(receipt):
    """Use original Q/A and verified quotes; titles are navigation, not facts."""
    data = receipt.result.get('request', {})
    company = data.get('company_id')
    if not company or receipt.user.company_id != company:
        return []
    root = f'a2-expert:{company}:{receipt.pk}'
    # Preserve explicitly marked sample provenance for the whole submission,
    # including semantic units whose short quote omits the original marker.
    is_demo = data.get('is_demo') is True or any(
        re.search(r'^\s*(?:\[DEMO\]|DEMO\s*:|Mẫu trả lời demo\s*:)', data.get(key, ''), re.I | re.M)
        for key in ('answer', 'note'))
    common = dict(company_id=company, submission_id=receipt.pk, document_id=root,
        contract_id=None, version_id=None, page_number=1, bbox=[], layout_type='expert_text',
        author=receipt.user.username, review_status='draft', is_demo=bool(is_demo),
        source_url=f'/newidea/?submission_id={receipt.pk}#expert-submission',
        source_location=f'Expert submission {receipt.pk}', original_name=f'Expert: {receipt.user.username}',
        heading_path=[], ancestor_ids=[], question=receipt.query)
    source_texts = {'question': receipt.query, 'expert-answer': data.get('answer', ''),
                    'expert-note': data.get('note', '')}
    records = []
    for source, text in source_texts.items():
        if text.strip():
            key = root+':'+source
            records.append(text_record(dict(common, knowledge_key=key, chunk_id=key,
                text=text, evidence_kind='expert_question' if source=='question' else 'expert_unit',
                source_role=source, heading_path=[receipt.query],
                graph_node_key=root if source=='question' else root+':answer' if source=='expert-answer' else root)))
    for unit in receipt.result.get('semantic_analysis', {}).get('units', []):
        source = unit.get('source_id', '').rsplit(':', 1)[0]
        quote = unit.get('quote', '')
        if not quote or source in source_texts and quote not in source_texts[source]:
            continue
        # Document quotes already have source pointers recorded by validation.
        if source not in source_texts and not unit.get('snapshot_key'):
            continue
        key = root+':'+unit['id']
        records.append(text_record(dict(common, knowledge_key=key, chunk_id=key, text=quote,
            evidence_kind='expert_question' if source=='question' else 'expert_unit',
            source_role=source, graph_node_key=key, unit_kind=unit['kind'],
            source_chunk_id=unit.get('source_chunk_id', ''),
            source_url=unit.get('source_url') or common['source_url'],
            heading_path=[receipt.query, unit['title']], concepts=unit.get('concepts', []))))
    return records


def visible_text_records(user):
    """Reload source records for access checks and stale-vector rejection."""
    jobs = expert_submission_repository.visible_submissions(user)
    records = [record for receipt in jobs for record in submission_records(receipt) if not record.get('is_demo')]
    for case in contract_repository.approved_visible_cases(user):
        if not evidence_current(case):
            continue
        details = serialize_case(case)
        if details['is_demo']:
            continue
        evidence = with_expert_evidence([], [details])[0]
        evidence.pop('evidence_id', None)
        records.append(text_record(dict(evidence, knowledge_key=evidence['chunk_id'],
            company_id=case.version.contract.company_id, document_id='a2-case:'+str(case.case_id),
            review_status=case.status, expert_case=details, page_number=1, bbox=[])))
    return records


def submission_context(receipt, records):
    """Source-bound saved graph edges, including ANSWERS and shared concepts."""
    if receipt.result.get('graph_index', {}).get('status') == 'cleared':
        return dict(state='cleared', version_id=None, nodes=[], edges=[],
                    origin='explicit_graph_reset')
    root = f'a2-expert:{receipt.result["request"]["company_id"]}:{receipt.pk}'
    nodes, edges = {}, []
    by_graph = {}
    for record in records:
        if record.get('submission_id') != receipt.pk:
            continue
        key = record['graph_node_key']
        by_graph.setdefault(key, []).append(record)
        node = nodes.setdefault(key, dict(id=key, kind=record.get('unit_kind', record['source_role']),
            label=record.get('text', '')[:100], source_leaf_ids=[]))
        node['source_leaf_ids'].append(record['chunk_id'])
    for relation in receipt.result.get('semantic_analysis', {}).get('relations', []):
        source = root if relation['from_id']=='question' else root+':'+relation['from_id']
        target = root if relation['to_id']=='question' else root+':'+relation['to_id']
        if source in by_graph and target in by_graph:
            edges.append(dict(from_id=source, to_id=target, type=relation['kind'], review_status='unreviewed',
                source_leaf_ids=[r['chunk_id'] for endpoint in (source, target) for r in by_graph[endpoint]]))
    return dict(state='ready', version_id=None, nodes=list(nodes.values()), edges=edges,
                origin='persisted_a2_graph_proposals', review_status='draft')


def contexts_for_records(records):
    identities = {r['submission_id'] for r in records if r.get('submission_id')}
    jobs = expert_submission_repository.submissions_by_ids(identities)
    return [submission_context(receipt, records) for receipt in jobs]

