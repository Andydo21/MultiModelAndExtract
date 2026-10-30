"""Freeze pilot50 inputs inside the web container: gold, production retrieval, Neo4j graphs, production pods.

Run: docker exec -e RAG_INFERENCE_URL=http://host.docker.internal:8010 riskdl_web \
     python -X utf8 -m evaluation.runners.pilot_freeze OUT_DIR
Retrieval is resumable (retrieval_partial.jsonl). Nothing is written to Neo4j/Qdrant.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
import django
django.setup()

from django.contrib.auth import get_user_model

from ai_extract.models import RetrievalSnapshot
from ai_extract.repositories.document_graph_repository import connection
from ai_extract.services.document_graph_service import graph_scope, read_context_graph
from NewIdea.services.knowledge_answer_service import answer_source_pods, bind_graph_context, render_source_pods
from NewIdea.services.knowledge_search_service import search_knowledge
from evaluation.io import digest

BASE = Path('research_evaluation_20261030')
USER_ID = 1000000  # corpus20_local; superuser sees every indexed version, as in the pilot draft
TOP_RETRIEVED = 10
REVIEWER = ('Claude Opus 5.5 — light check: question/reference/tuples compared against gold evidence text; '
            'not an independent human review; accepted by project owner 2026-10-30')
snapshots = {}


def snapshot(version):
    if version not in snapshots:
        snapshots[version] = RetrievalSnapshot.objects.select_related('run__version__contract').get(version_id=version)
    return snapshots[version]


def leaf(version, chunk):
    return next(x for x in snapshot(version).tree['leaves'] if x['chunk_id'] == chunk)


def table_of(source):
    """Same acceptance rule as production pods: cells must reproduce the OCR text exactly."""
    metadata = source.get('structure_metadata') or {}
    rows = metadata.get('table_rows')
    if metadata.get('table_structure_status') in {'unresolved', 'unavailable', 'rejected', 'failed'}:
        return None
    if not (isinstance(rows, list) and len(rows) > 1 and all(isinstance(r, list) for r in rows)):
        return None
    flat = ' '.join(str(c) for r in rows for c in r if c is not None)
    if re.sub(r'\s+', '', flat) != re.sub(r'\s+', '', source.get('text', '')):
        return None
    width = max(len(r) for r in rows)
    pad = [[('' if c is None else str(c)) for c in r] + [''] * (width - len(r)) for r in rows]
    return dict(columns=pad[0], rows=pad[1:])


def evidence(version, chunk):
    run = snapshot(version).run_id
    eid = f'a3:v{version}:r{run}:{chunk}'
    source = dict(leaf(version, chunk), evidence_id=eid)
    return dict(evidence_id=eid, text=source.get('text', ''), document_id=source['document_id'], version=version,
        page=source['page_number'], region=source.get('bbox'), breadcrumb=source.get('heading_path', []),
        scope=dict(document_id=source['document_id'], version_id=version, heading_path=source.get('heading_path', [])),
        links=[], layout_type=source.get('layout_type'), content_hash=source.get('content_hash'),
        original_name=source.get('original_name'), table=table_of(source), production_source=source)


def gold(out):
    questions = [json.loads(l) for l in (BASE/'pilot50_current_questions.jsonl').open(encoding='utf8')]
    old = {e['evidence_id']: e for e in map(json.loads, (BASE/'pilot50_current_evidence.jsonl').open(encoding='utf8'))}
    nodes = {}
    for eid in old:
        version, chunk = old[eid]['version'], old[eid]['production_source']['chunk_id']
        node = evidence(version, chunk)
        if node['evidence_id'] != eid or node['text'] != old[eid]['text']:
            raise ValueError(f'Gold evidence drifted from current snapshot: {eid}')
        nodes[eid] = node
    with (out/'gold_questions.jsonl').open('x', encoding='utf8') as stream:
        for q in questions:
            q = dict(q, review_status='approved', reviewer=REVIEWER, review_scope='light_text_check',
                     review_note='Values checked against gold evidence text only; page images not re-inspected.')
            stream.write(json.dumps(q, ensure_ascii=False) + '\n')
    return questions, nodes


def retrieve(out, questions, nodes):
    user = get_user_model().objects.get(pk=USER_ID)
    partial = out/'retrieval_partial.jsonl'
    done = {json.loads(l)['question_id']: json.loads(l) for l in partial.open(encoding='utf8')} if partial.exists() else {}
    gold_hash = {n['content_hash']: n['evidence_id'] for n in nodes.values()}
    for q in questions:
        if q['question_id'] in done:
            continue
        started = time.perf_counter()
        result = search_knowledge(user, q['question'], generate=False)
        latency = (time.perf_counter() - started) * 1000
        ranked = [c for c in result['citations'] if c.get('expansion_reason') == 'query_match']
        items = []
        for rank, c in enumerate(ranked[:TOP_RETRIEVED], 1):
            node = evidence(c['version_id'], c['chunk_id'])
            nodes.setdefault(node['evidence_id'], node)
            items.append(dict(rank=rank, evidence_id=node['evidence_id'], rrf_score=c.get('rrf_score'),
                neural_logit=c.get('neural_logit'), streams=c.get('retrieval_streams'),
                gold_equivalent_of=gold_hash.get(node['content_hash'])))
        row = dict(question_id=q['question_id'], question_sha256=digest(q['question']), latency_ms=latency,
            ranked=items, query_match_total=len(ranked), citations_total=len(result['citations']),
            expanded_reasons=sorted({c.get('expansion_reason') for c in result['citations']}),
            streams=(result.get('context_coverage') or {}).get('retrieval_streams'))
        with partial.open('a', encoding='utf8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
        done[q['question_id']] = row
        print('retrieved', q['question_id'], round(latency), len(items), flush=True)
    return done


def export_graphs(versions):
    graphs = []
    with connection() as session:
        for version in sorted(versions):
            snap = snapshot(version)
            scope = graph_scope(snap.run, snap.tree)
            nodes = session.run('MATCH (n:A3Node {snapshot_key:$k}) RETURN properties(n) AS n', k=scope['key']).data()
            edges = session.run('MATCH (a:A3Node {snapshot_key:$k})-[r]->(b:A3Node {snapshot_key:$k}) '
                                'RETURN a.key AS source, b.key AS target, type(r) AS type', k=scope['key']).data()
            graphs.append(dict(version_id=version, run_id=snap.run_id, scope=scope,
                nodes=[{k: v for k, v in r['n'].items() if k not in ('identity_embedding', 'embedding')} for r in nodes],
                edges=edges, leaves=[l['chunk_id'] for l in snap.tree['leaves']]))
            print('graph', version, len(nodes), len(edges), flush=True)
    return graphs


def production_pods(question, ids, nodes):
    sources = [nodes[i]['production_source'] for i in ids]
    contexts = []
    for version in sorted({s['version_id'] for s in sources}):
        snap = snapshot(version)
        contexts.append(read_context_graph(snap.run, snap.tree, [s for s in sources if s['version_id'] == version]))
    graph = bind_graph_context(contexts, sources, query=question)
    pods = answer_source_pods(sources, graph)
    order = [s['id'] for pod in pods['pods'] for s in pod['sources']]
    return dict(order=order, rendered=render_source_pods(pods), links=len(pods['source_links']),
                context_states=[c.get('state') for c in contexts])


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    questions, nodes = gold(out) if not (out/'gold_questions.jsonl').exists() else (
        [json.loads(l) for l in (out/'gold_questions.jsonl').open(encoding='utf8')], None)
    if nodes is None:
        old = {e['evidence_id']: e for e in map(json.loads, (BASE/'pilot50_current_evidence.jsonl').open(encoding='utf8'))}
        nodes = {eid: evidence(e['version'], e['production_source']['chunk_id']) for eid, e in old.items()}
    retrieval = retrieve(out, questions, nodes)
    pods = {}
    for q in questions:
        for mode, ids in (('gold', q['gold_evidence_ids']),
                          ('retrieved', [x['evidence_id'] for x in retrieval[q['question_id']]['ranked'][:5]])):
            pods[q['question_id']+':'+mode] = production_pods(q['question'], ids, nodes)
    versions = {n['version'] for n in nodes.values()}
    (out/'graphs.json').write_text(json.dumps(export_graphs(versions), ensure_ascii=False, default=str), encoding='utf8')
    (out/'production_pods.json').write_text(json.dumps(pods, ensure_ascii=False), encoding='utf8')
    with (out/'evidence.jsonl').open('w', encoding='utf8') as stream:
        for node in nodes.values():
            stream.write(json.dumps(node, ensure_ascii=False, default=str) + '\n')
    (out/'retrieval.json').write_text(json.dumps(dict(retriever='production search_knowledge(generate=False): BM25 + BGE-M3 dense + ColPali, RRF',
        reranker_actual='BAAI/bge-reranker-v2-m3 (name reported by local runtime)', user_id=USER_ID,
        top_k=TOP_RETRIEVED, questions=retrieval), ensure_ascii=False, default=str), encoding='utf8')
    print('frozen', len(questions), 'questions', len(nodes), 'evidence nodes', len(versions), 'versions')


def export_leaves(out):
    """Every source leaf of the frozen graph versions, so E2 neighbours carry real text."""
    out = Path(out)
    graphs = json.loads((out/'graphs.json').read_text(encoding='utf8'))
    with (out/'graph_leaves.jsonl').open('x', encoding='utf8') as stream:
        for graph in graphs:
            for chunk in graph['leaves']:
                node = evidence(graph['version_id'], chunk)
                stream.write(json.dumps(node, ensure_ascii=False, default=str) + '\n')
    print('leaves', sum(len(g['leaves']) for g in graphs))


if __name__ == '__main__':
    export_leaves(sys.argv[1]) if sys.argv[2:] == ['--leaves'] else main(sys.argv[1])
