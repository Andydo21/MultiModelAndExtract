"""Search the authorized knowledge corpus using BM25, A3 fusion and reranking."""
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from threading import Lock
import time
from django.conf import settings
from ai_extract.services.contract_pipeline_service import HybridBackend, hydrate, expand_retrieval_context, plan_retrieval_questions
from shared.retrieval import LexicalOnlyReranker, evidence_identity, identifiers
from NewIdea.repositories import contract_repository, knowledge_repository
from .search_job_service import reserve_search, finish_search, update_progress
from .errors import KnowledgeError
from .search_evidence_service import (
    matched_structure_nodes, select_answer_sources, select_retrieval_seeds,
    visible_text_records, contexts_for_records, visual_source_context,
)

def refresh_snapshot(version):
    """Build the source tree in the service; repository persists its index atomically."""
    from django.db import transaction
    from ai_extract.services.document_structure_service import structure_for_version
    from ai_extract.services.document_graph_service import graph_scope
    with transaction.atomic():
        snapshot,version=contract_repository.lock_snapshot(version)
        run=contract_repository.completed_extraction_run(version)
        records=contract_repository.version_sources(version)
        tree=structure_for_version(version,records,version.contract.title)
        scope=graph_scope(run,tree) if run and tree.get('graph_index',{}).get('status')=='ready' else {}
        return contract_repository.replace_snapshot(snapshot,version,run,records,tree,scope)


def indexed_corpus(versions, runs, knowledge):
    snapshots=contract_repository.retrieval_snapshots(versions)
    rebuilt=0
    for version in versions:
        snapshot=snapshots.get(version.pk)
        if (snapshot is None or snapshot.dirty or snapshot.schema!=1
            or snapshot.run_id!=getattr(runs.get(version.pk),'pk',None)
            or snapshot.metadata_key!=contract_repository.metadata_key(version)):
            snapshots[version.pk]=refresh_snapshot(version)
            rebuilt+=1
    return contract_repository.IndexedCorpus(snapshots,knowledge,rebuilt)


class SearchBackend(HybridBackend):
    def __init__(self):
        self.embeddings = {}
        self.visual_embeddings = {}
        self.embedding_lock = Lock()
        self.visual_lock = Lock()

    @staticmethod
    def _cached(cache, key, lock, operation, argument):
        # Shared by parallel document searches. Store failures too so an
        # unavailable model is not retried once for every selected document.
        with lock:
            if key not in cache:
                try:
                    cache[key] = operation(argument)
                except Exception as error:
                    cache[key] = error
            result = cache[key]
        if isinstance(result, Exception):
            raise result
        return result

    def embedding(self, texts):
        return self._cached(self.embeddings, tuple(texts), self.embedding_lock, super().embedding, texts)

    def colpali_embedding(self, query):
        return self._cached(self.visual_embeddings, query, self.visual_lock, super().colpali_embedding, query)

    def prime_embeddings(self, queries):
        """One BGE batch for the question and all model-planned parts."""
        with self.embedding_lock:
            pending=[q for q in dict.fromkeys(queries) if (q,) not in self.embeddings]
            if not pending:return
            try:
                vectors=super().embedding(pending)
                if len(vectors)!=len(pending):raise ValueError('Incomplete query embedding batch')
            except Exception as error:
                for query in pending:self.embeddings[(query,)]=error
            else:
                for query,vector in zip(pending,vectors):self.embeddings[(query,)]=[vector]



def collect_questions(queries, collect):
    """Bound corpus I/O, preserve question order and close worker DB connections."""
    workers = min(2, max(1, int(getattr(settings, 'A2_SEARCH_WORKERS', 1))))
    if workers == 1 or len(queries) < 2:
        return [(q, *collect(q)) for q in queries]
    def run(query):
        from django.db import connections, close_old_connections
        close_old_connections()
        try:
            return (query, *collect(query))
        finally:
            connections.close_all()
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='a2-questions') as pool:
        return list(pool.map(run, queries))

def search_knowledge(user, query, generate=False, existing_receipt=None):
    """Search stored knowledge once per vector space, then rerank all sources together."""
    started = time.perf_counter()
    timings = {}
    checkpoint = started
    def mark(stage):
        nonlocal checkpoint
        now = time.perf_counter()
        timings[stage] = round((now-checkpoint)*1000, 2)
        checkpoint = now
    receipt = existing_receipt
    if receipt is not None:
        if receipt.user_id != user.pk or receipt.version_id is not None or receipt.scope not in {'knowledge', 'company'} or receipt.query != query.strip():
            raise KnowledgeError('Search job scope mismatch.', 403)
    else:
        receipt = reserve_search(user, query, scope='knowledge', generated=generate)
    if receipt is None:
        raise KnowledgeError('Search quota reached; retry after one minute.', 429)
    try:
        query = query.strip()
        update_progress(receipt, 'retrieving')
        versions, total = contract_repository.latest_visible_versions(user, indexed_only=False, limit=None, load_sources=False)
        by_version = {v.pk: v for v in versions}
        dense_runs, visual_runs, runs = {}, {}, {}
        for version in versions:
            run = contract_repository.completed_extraction_run(version)
            if run:
                runs[version.pk] = run
                if run.dense_index_status == 'COMPLETED':
                    dense_runs[version.pk] = run.pk
                if run.colpali_index_status == 'COMPLETED':
                    visual_runs[version.pk] = run.pk
        knowledge = visible_text_records(user)
        by_key = {r['knowledge_key']: r for r in knowledge}
        # Keyword lookup is one corpus stream, not a separate expert-case path.
        trees = indexed_corpus(versions, runs, knowledge)
        mark('corpus_index')
        backend = SearchBackend()
        def collect(query):
            streams = {'keyword': trees.keyword_search(query, limit=60)}
            diagnostics = {'keyword': dict(state='ready', candidates=len(streams['keyword']))}
            if getattr(settings, 'A3_HYBRID_ENABLED', True):
                import qdrant_client.models  # Avoid concurrent cold-import cycles.
                with ThreadPoolExecutor(max_workers=2, thread_name_prefix='a2-corpus') as pool:
                    futures = {}
                    if dense_runs or knowledge:
                        futures['dense'] = pool.submit(backend.dense_knowledge, query, versions, dense_runs, records=knowledge)
                    if visual_runs:
                        futures['colpali'] = pool.submit(backend.colpali_knowledge, query, versions, visual_runs)
                    for name, task in futures.items():
                        try:
                            hits = task.result()
                            selected = []
                            document_hits = {}
                            for hit in hits:
                                payload = hit.payload or {}
                                if payload.get('source_kind') == 'knowledge_text':
                                    record = by_key.get(payload.get('knowledge_key'))
                                    if (name == 'dense' and record and payload.get('source_app') == 'contract_dl'
                                        and payload.get('company_id') == record['company_id']
                                        and payload.get('content_sha256') == record['content_sha256']):
                                        selected.append(dict(record, vector_rank_score=float(hit.score)))
                                    continue
                                version = by_version.get(payload.get('version_id'))
                                expected = dense_runs if name == 'dense' else visual_runs
                                if (version and version.pk in expected and payload.get('source_app') == 'contract_dl'
                                    and payload.get('company_id') == version.contract.company_id
                                    and payload.get('contract_id') == version.contract_id
                                    and payload.get('contract_dl_run_id') == expected.get(version.pk)):
                                    document_hits.setdefault(version.pk, []).append(hit)
                            for identity, points in document_hits.items():
                                selected.extend(hydrate(points, trees[identity]['leaves'], by_version[identity],
                                    runs[identity].pk, visual=name=='colpali', query=query))
                            # Preserve the global figure rank across documents and same-page images.
                            if name == 'colpali':
                                page_ranks = {(h.payload.get('version_id'), h.payload.get('page_number'), h.payload.get('visual_block_id')): i
                                              for i, h in reversed(list(enumerate(hits, 1)))}
                                selected = [dict(r, colpali_page_rank=page_ranks.get((r['version_id'], r['page_number'], r.get('visual_block_id')), r['colpali_page_rank']))
                                            for r in selected]
                            else:
                                selected.sort(key=lambda r: r.get('vector_rank_score', 0), reverse=True)
                            streams[name] = selected
                            diagnostics[name] = dict(state='ready' if selected else 'empty', candidates=len(selected))
                        except Exception as error:
                            diagnostics[name] = dict(state='unavailable', error_type=type(error).__name__, candidates=0)
            for name in ('dense', 'colpali'):
                diagnostics.setdefault(name, dict(state='not_indexed' if getattr(settings, 'A3_HYBRID_ENABLED', True) else 'disabled', candidates=0))
            candidates = LexicalOnlyReranker.reciprocal_rank_fusion(streams)
            requested = identifiers(query)
            matched_documents = trees.identifier_documents(requested)
            if matched_documents:
                candidates = [r for r in candidates if r.get('document_id') in matched_documents]
            candidate_count = len(candidates)
            candidates = candidates[:max(1,int(getattr(settings,'A2_RERANK_CANDIDATES',20)))]
            return candidates,diagnostics,matched_documents,candidate_count
        planning = plan_retrieval_questions(query) if generate else dict(status='not_requested',queries=[query])
        mark('question_planning')
        search_queries=list(dict.fromkeys([query,*planning['queries']]))
        if getattr(settings,'A3_HYBRID_ENABLED',True) and (dense_runs or knowledge):
            backend.prime_embeddings(search_queries)
        mark('query_embedding')
        searches=collect_questions(search_queries, collect)
        # All embedding lookups precede reranking: keep each GPU model resident
        # for its stage instead of swapping models for every subquestion.
        mark('candidate_search')
        merged, question_results, source_hints, source_ranks = {}, [], {}, {}
        reranker_state='disabled'
        candidate_count=0
        reranked = None
        rerank_failed = False
        if getattr(settings, 'A3_HYBRID_ENABLED', True) and any(s[1] for s in searches):
            update_progress(receipt, 'reranking', candidates=sum(len(s[1]) for s in searches))
            try:
                # A facet such as a temperature or deadline is not an independent
                # subject. Keep the original request in every reranker pair so
                # a locally relevant passage about another entity ranks lower.
                reranked = backend.rerank_many([
                    (query if s[0] == query else f'{query}\nFocused question: {s[0]}', s[1])
                    for s in searches])
            except Exception:
                rerank_failed = True
        for search_index, (search_query,ranked,current,documents,count) in enumerate(searches):
            candidate_count+=count
            if search_query==query:
                diagnostics,matched_documents=current,documents
            state='disabled'
            if ranked and getattr(settings,'A3_HYBRID_ENABLED',True):
                update_progress(receipt,'reranking',candidates=len(ranked))
                if not rerank_failed:
                    ranked=reranked[search_index]
                    state='ready'
                else:
                    state='unavailable'
                    if generate:
                        ranked=[]
            if state=='ready' or reranker_state!='ready':
                reranker_state=state
            question_results.append(dict(query=search_query,reranker_state=state,eligible_candidates=len(ranked)))
            for rank,source in enumerate(ranked,1):
                identity=evidence_identity(source)
                source_ranks.setdefault(identity, []).append(dict(query=search_queries.index(search_query),rank=rank))
                if generate and search_query in planning['queries']:
                    parts=planning.get('answer_parts') or [dict(question=q,retrieval_query=q) for q in planning['queries']]
                    source_hints.setdefault(identity,[]).extend(dict(part=i,rank=rank) for i,part in enumerate(parts,1)
                        if part['retrieval_query']==search_query)
                previous=merged.get(identity)
                if previous is None or source.get('neural_logit',float('-inf'))>previous.get('neural_logit',float('-inf')):
                    merged[identity]=dict(source)
                if identity in source_hints:
                    merged[identity]['retrieval_hints']=source_hints[identity]
                merged[identity]['retrieval_ranks']=source_ranks[identity]
        mark('reranking')
        # Deduplicate before admission, then select across all subquestions.
        # Only admitted seeds may expand graph context; discarded documents do
        # not consume graph reads or prompt space.
        seed_sources=list(merged.values())
        from .knowledge_answer_service import eligible_answer_sources
        seed_sources=eligible_answer_sources(seed_sources)
        eligible=select_answer_sources(seed_sources,limit=len(seed_sources))
        seed_limit=getattr(settings,'A2_ANSWER_SOURCE_LIMIT',24)
        if generate:
            # Atomic searches improve coverage without multiplying graph expansion.
            groups=1+planning.get('seed_group_count',len(planning['queries']))
            parts=len(planning.get('answer_parts') or planning['queries'])
            seed_limit=min(seed_limit,max(parts,max(1,int(getattr(settings,'A2_SEEDS_PER_QUESTION',3)))*groups))
        candidates=select_retrieval_seeds(eligible,limit=seed_limit)
        identity_discovery=dict(state='disabled',candidates=0,policy='document_local_graphs_only')
        for item in question_results:
            ordinal=search_queries.index(item['query'])
            item['selected_seeds']=sum(any(h['query']==ordinal for h in c.get('retrieval_ranks',[])) for c in candidates)
        # Only context expansion is document-specific. No second vector lookup.
        candidates_by_version=defaultdict(list)
        for candidate in candidates:
            if candidate.get('source_kind')!='knowledge_text':
                candidates_by_version[candidate.get('version_id')].append(candidate)
        selected_versions = [v for v in versions if v.pk in candidates_by_version]
        details, expanded, matched_nodes = [], [], []
        update_progress(receipt, 'reading_graph', documents_total=len(selected_versions))
        for index, version in enumerate(selected_versions, 1):
            ranked = candidates_by_version[version.pk]
            context, coverage = expand_retrieval_context(query, trees[version.pk], version, ranked,
                dict(retrieval_mode='unified_corpus', retrieval_streams=diagnostics,
                     reranker_state=reranker_state, rerank_stage='shared_corpus', fusion='equal_weight_rrf'),backend=backend)
            expanded.extend(c for c in context if c['expansion_reason'] != 'query_match')
            expanded.extend(visual_source_context(ranked, trees[version.pk]['leaves']))
            details.append(dict(version_id=version.pk, coverage=coverage))
            update_progress(receipt, 'reading_graph', documents_done=index, documents_total=len(selected_versions))
        # Reuse complete Q/A context for a matched unit, preserving its conditions.
        expert_roots = {c['submission_id'] for c in candidates if c.get('submission_id')}
        expanded.extend(dict(r, expansion_reason='expert_submission_context') for r in knowledge
            if r.get('submission_id') in expert_roots and r.get('source_role') in {'question', 'expert-answer', 'expert-note'})
        results = select_answer_sources([dict(c, expansion_reason='query_match') for c in candidates] + expanded)
        citations = results
        ids={(c.get('version_id'),c.get('chunk_id')):c['evidence_id'] for c in citations}
        for citation in citations:
            links=[]
            for link in citation.get('visual_context_links',[]):
                figure_id=ids.get((citation.get('version_id'),link.get('figure_chunk_id')))
                if figure_id:
                    links.append(dict(figure_evidence_id=figure_id, kind=link['kind'],
                        spatial_relation=link.get('spatial_relation'),
                        review_status='source_supplied' if link['kind']=='surya_child' else 'unverified_candidate'))
            if links:citation['visual_context']=links
        for version in selected_versions:
            matched_nodes.extend(matched_structure_nodes(trees[version.pk],
                [c for c in citations if c.get('version_id') == version.pk and c.get('source_kind') != 'knowledge_text'], version, runs.get(version.pk)))
        for citation in citations:
            if citation.get('source_kind') == 'knowledge_text':
                matched_nodes.append(dict(node_id=citation.get('graph_node_key', citation['chunk_id']),
                    title=citation.get('text', '')[:100], kind=citation.get('unit_kind', 'expert_text'),
                    evidence_ids=[citation['evidence_id']], source_url=citation['source_url'],
                    retrieval_basis='unified_knowledge_hit', review_status=citation.get('review_status')))
        graph_contexts = contexts_for_records(
            [r for r in knowledge if r.get('submission_id') in expert_roots])
        cases = list({c['expert_case_id']: c['expert_case'] for c in citations if c.get('expert_case_id')}.values())
        result = dict(success=True, results=results, citations=citations, matched_nodes=matched_nodes[:30],
            similar_cases=cases, search_id=receipt.pk,
            generation_status='discovery_sources', score_kind='retrieval_ranking_not_accuracy',
            context_coverage=dict(mode='unified_knowledge', access_scope='all_companies' if user.is_superuser else 'authorized_documents',
                versions_total=total, versions_scanned=len(versions), indexed_only=False,
                versions_with_sources=sum(bool(s.passage_count) for s in trees.snapshots.values()),
                retrieval_index=dict(rebuilt_versions=trees.rebuilt, hydrated_versions=len(trees.loaded)),
                indexed_versions=len(set(dense_runs)|set(visual_runs)),
                query_planning=planning, requested_parts=[p['question'] for p in planning.get('answer_parts',[])] or planning['queries'],
                answer_parts=planning.get('answer_parts',[]),retrieval_questions=question_results,
                source_selection=dict(policy='part_coverage_ranked_novelty_before_graph',
                    eligible_seeds=len(eligible),selected_seeds=len(candidates),seed_limit=seed_limit,
                    omitted_seeds=len(eligible)-len(candidates)),
                shared_entity_discovery=identity_discovery,
                versions_deeply_searched=len(selected_versions), latest_versions_only=True,
                identifier_matched_versions=[v.pk for v in versions if f'contract-version-{v.pk}' in matched_documents],
                partial=candidate_count>max(1,int(getattr(settings,'A2_RERANK_CANDIDATES',20))) or len(versions)<total or len(eligible)>len(candidates), retrieval_streams=diagnostics,
                dense_state=diagnostics['dense']['state'], reranker_state=reranker_state,
                knowledge_records=len(knowledge), version_details=details,
                graph_contexts=graph_contexts, external_web_search=False))
        mark('source_and_graph_expansion')
        result['retrieval_latency_ms'] = round((time.perf_counter()-started)*1000, 2)
        if generate and reranker_state=='unavailable' and not candidates:
            result.update(answer='Dịch vụ tìm kiếm nguồn tạm không sẵn sàng. Chưa thể kết luận tài liệu thiếu thông tin; vui lòng thử lại khi dịch vụ hoạt động.',
                generation_status='retrieval_unavailable',generation_diagnostics=dict(state='retrieval_unavailable'))
        elif generate:
            from .knowledge_answer_service import answer_from_sources
            result.update(answer_from_sources(query, citations, result['context_coverage'],
                progress=lambda stage: update_progress(receipt, stage)))
        mark('answer_generation')
        result['stage_timings_ms'] = timings
        result['latency_ms'] = round((time.perf_counter()-started)*1000, 2)
        finish_search(receipt, result)
        return result
    finally:
        if receipt.status == 'RUNNING':
            receipt.status = 'FAILED'
            knowledge_repository.save_search(receipt, 'status')
