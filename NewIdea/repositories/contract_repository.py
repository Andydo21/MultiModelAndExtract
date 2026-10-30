"""Authorized contracts, versions and persisted A3 evidence."""
from django.db.models import Exists, OuterRef, Prefetch, Q, Subquery
from contracts.models import Contract, ContractVersion, ContractKnowledgeCase
from ai_extract.models import LayoutExtractionRun
from django.urls import reverse


def latest_visible_versions(user, indexed_only=False, *, limit=100, load_sources=True):
    latest = ContractVersion.objects.filter(contract_id=OuterRef('contract_id')).order_by('-version_number', '-pk').values('pk')[:1]
    versions = ContractVersion.objects.filter(contract__in=visible_contracts(user),
        pk=Subquery(latest)).select_related('contract').order_by('-created_at', '-pk')
    total = versions.count()
    if indexed_only:
        indexed_runs = LayoutExtractionRun.objects.filter(version_id=OuterRef('pk'), status='COMPLETED').filter(
            Q(dense_index_status='COMPLETED') | Q(colpali_index_status='COMPLETED'))
        versions = versions.filter(Exists(indexed_runs))
    runs = LayoutExtractionRun.objects.filter(status='COMPLETED')
    if load_sources:
        runs = runs.prefetch_related('pages__blocks')
        versions = versions.prefetch_related('ai_extract_clauses', 'contexts')
    else:
        runs = runs.defer('structure', 'error_message', 'dense_index_error', 'colpali_index_error')
    versions = versions.prefetch_related(Prefetch('layout_extraction_runs', queryset=runs))
    return list(versions[:limit] if limit is not None else versions), total


def get_visible_version(user, version_id):
    return ContractVersion.objects.filter(contract__in=visible_contracts(user)).get(pk=version_id)


def version_sources(version):
    return source_records(version)


def completed_extraction_run(version):
    cached = getattr(version, '_prefetched_objects_cache', {}).get('layout_extraction_runs')
    if cached is not None:
        return next((run for run in cached if run.status == 'COMPLETED'), None)
    return version.layout_extraction_runs.filter(status='COMPLETED').first()


def approved_visible_cases(user):
    return list(case_queryset(user).filter(status='approved'))

def visible_contracts(user):
    if not user.is_authenticated:
        return Contract.objects.none()
    if user.is_superuser:
        return Contract.objects.all()
    if not user.company_id:
        return Contract.objects.none()
    return Contract.objects.filter(company_id=user.company_id)


def case_queryset(user):
    return ContractKnowledgeCase.objects.filter(version__contract__in=visible_contracts(user)).select_related(
        'version__contract', 'author', 'reviewer')


def source_records(version):
    """Prefer verified page blocks; preserve clause/context provenance as fallback."""
    records = []
    contract = version.contract
    common = dict(document_id=f'contract-version-{version.pk}', contract_id=contract.pk,
                  version_id=version.pk, version_number=version.version_number,
                  original_name=f'{contract.contract_code} — {contract.title}')
    run = completed_extraction_run(version)
    if run:
        pages = run.pages.all() if 'pages' in getattr(run, '_prefetched_objects_cache', {}) else run.pages.prefetch_related('blocks')
        for page in pages:
            for block in page.blocks.all():
                if block.text.strip() or (block.structure_metadata or {}).get('image_path') or block.layout_type == 'figure':
                    records.append(dict(common, chunk_id=f'block:{block.pk}', text=block.text,
                        page_number=page.page_number, bbox=block.bbox, layout_type=block.layout_type,
                        structure_metadata=block.structure_metadata, image_path=page.image_path,
                        source_location=(block.structure_metadata or {}).get('source_location') or f'Phiên bản {version.version_number}, trang {page.page_number}',
                        source_url=reverse('ai_extract_page', args=[contract.pk, page.page_number]) + f'?version_id={version.pk}'))
    for clause in (version.ai_extract_clauses.all() if not records else []):
        records.append(dict(common, chunk_id=f'clause:{clause.pk}',
            text=f'{clause.clause_title}\n{clause.clause_content}', layout_type='clause', bbox=[],
            source_location=f'Phiên bản {version.version_number}, điều khoản {clause.clause_title}',
            source_url=reverse('contract_detail', args=[contract.pk]) + f'?version_id={version.pk}'))
    if not records:
        cached = getattr(version, '_prefetched_objects_cache', {}).get('contexts')
        contexts = (context for context in cached if context.context_type == 'raw_text') if cached is not None else version.contexts.filter(context_type='raw_text')
        for context in contexts:
            records.append(dict(common, chunk_id=f'context:{context.pk}', text=context.content,
                layout_type='context', bbox=[], source_location=f'Phiên bản {version.version_number}, {context.source}',
                source_url=reverse('contract_detail', args=[contract.pk]) + f'?version_id={version.pk}'))
    return records



def get_visible_contract(user, contract_id):
    return visible_contracts(user).get(pk=contract_id)


def contract_version(contract, version_id=None):
    return contract.versions.get(pk=version_id) if version_id else contract.latest_version


def approved_related_cases(user, contract):
    cases = case_queryset(user).filter(status='approved')
    return list(cases.filter(version__contract__contract_type=contract.contract_type) if contract.contract_type else cases.filter(version__contract=contract))


def workspace_contracts(user):
    return list(visible_contracts(user).order_by('-updated_at'))


# Derived retrieval data stays in the repository; source facts remain in A3.
def metadata_key(version):
    return repr((version.contract_id, version.contract.company_id,
                 version.contract.contract_code, version.contract.title, version.version_number))


def lock_snapshot(version):
    from ai_extract.models import RetrievalSnapshot
    snapshot,_=RetrievalSnapshot.objects.get_or_create(version=version)
    snapshot=RetrievalSnapshot.objects.select_for_update().get(pk=snapshot.pk)
    version=ContractVersion.objects.select_related('contract').get(pk=version.pk)
    return snapshot,version


def replace_snapshot(snapshot, version, run, records, tree, scope):
    from collections import Counter
    from ai_extract.models import RetrievalPassage
    from shared.retrieval import tokens
    rows=[]
    for record in tree['leaves']:
        if record.get('retrieval_excluded'):continue
        counts=dict(Counter(tokens(record.get('text','')+' '+record.get('original_name',''))))
        rows.append(RetrievalPassage(snapshot=snapshot,record=record,
                                    token_counts=counts,length=sum(counts.values())))
    snapshot.passages.all().delete()
    RetrievalPassage.objects.bulk_create(rows,batch_size=500)
    snapshot.run=run
    snapshot.records,snapshot.tree=records,tree
    snapshot.graph_scope=scope
    snapshot.passage_count,snapshot.token_count=len(rows),sum(r.length for r in rows)
    snapshot.metadata_key,snapshot.schema,snapshot.dirty=metadata_key(version),1,False
    snapshot.save()
    return snapshot


def retrieval_snapshots(versions):
    from ai_extract.models import RetrievalSnapshot
    return {s.version_id:s for s in RetrievalSnapshot.objects.filter(
        version_id__in=[v.pk for v in versions]).defer('records','tree')}


class IndexedCorpus:
    """Search authorized postings; hydrate full trees only for admitted documents."""
    def __init__(self, snapshots, knowledge, rebuilt=0):
        from collections import Counter
        from shared.retrieval import tokens
        from threading import Lock
        self.load_lock = Lock()
        self.loaded,self.rebuilt={},rebuilt
        self.snapshots=snapshots
        self.knowledge = []
        for record in knowledge:
            if not record.get('retrieval_excluded'):
                counts = dict(Counter(tokens(record.get('text', '') + ' ' + record.get('original_name', ''))))
                self.knowledge.append((record, counts, sum(counts.values())))
        self.count = sum(s.passage_count for s in self.snapshots.values()) + len(self.knowledge)
        self.average = (sum(s.token_count for s in self.snapshots.values()) + sum(r[2] for r in self.knowledge)) / max(1, self.count) or 1

    def __getitem__(self, version_id):
        with self.load_lock:
            if version_id not in self.loaded:
                self.loaded[version_id] = self.snapshots[version_id].tree
            return self.loaded[version_id]

    def _matching(self, terms):
        from ai_extract.models import RetrievalPassage
        if not terms:
            return []
        return list(RetrievalPassage.objects.filter(snapshot_id__in=[s.pk for s in self.snapshots.values()],
            token_counts__has_any_keys=list(terms)).order_by('pk').values_list('record', 'token_counts', 'length')) + [
                r for r in self.knowledge if terms.intersection(r[1])]

    def keyword_search(self, query, limit=60):
        from shared.retrieval import tokens, rank_indexed_records
        terms=set(tokens(query))
        return rank_indexed_records(terms,self._matching(terms),self.count,self.average,limit)

    def identifier_documents(self, requested):
        return {r['document_id'] for r, counts, _ in self._matching(set(requested))
                if set(requested).issubset(counts)}
