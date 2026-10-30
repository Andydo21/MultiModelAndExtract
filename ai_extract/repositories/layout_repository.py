from decimal import Decimal
"""Read extraction runs, page blocks and contract files; persist A3 extraction results."""
from ai_extract.models import LayoutExtractionRun, LayoutExtractionPage, LayoutExtractionBlock
from NewIdea.repositories.contract_repository import visible_contracts
from django.db import transaction
from django.db.models import OuterRef, Prefetch, Subquery
from contextlib import contextmanager
from threading import Event, Thread, local
import logging

WORKFLOW_LOCK_KEY=1093870915
WORKFLOW_HEARTBEAT_SECONDS=20
_workflow_state=local()


@contextmanager
def workflow_slot():
    """Serialize extraction/identity on the shared GPU; crashes release the lease.

    Use a dedicated transaction, which also works behind transaction-pooling
    proxies; session advisory locks are unsafe with that pooling mode.
    """
    from django.db import connections
    database=connections['default']
    if database.vendor!='postgresql':
        yield True
        return
    ready,stop=Event(),Event()
    state=dict(acquired=False,error=None)
    def maintain_lease():
        # This thread owns the connection from creation through close; Django
        # connections are never shared concurrently across threads.
        lease=None
        try:
            lease=database.copy(alias='a3_workflow_lease')
            lease.settings_dict=dict(lease.settings_dict,OPTIONS=dict(
                lease.settings_dict.get('OPTIONS',{}),connect_timeout=5,
                keepalives=1,keepalives_idle=10,keepalives_interval=5,keepalives_count=2,
                tcp_user_timeout=10000))
            lease.set_autocommit(False)
            with lease.cursor() as cursor:
                cursor.execute("SET LOCAL idle_in_transaction_session_timeout = '0'")
                cursor.execute("SET LOCAL statement_timeout = '5s'")
                cursor.execute('SELECT pg_try_advisory_xact_lock(%s)',[WORKFLOW_LOCK_KEY])
                state['acquired']=bool(cursor.fetchone()[0])
            ready.set()
            if state['acquired']:
                while not stop.wait(WORKFLOW_HEARTBEAT_SECONDS):
                    with lease.cursor() as cursor:cursor.execute('SELECT 1')
        except Exception as error:
            state['error']=error
            ready.set()
        finally:
            # A lost connection during cleanup cannot turn already committed
            # extraction results into a spurious API/batch failure.
            if lease is not None:
                try:lease.rollback()
                except Exception:
                    logging.getLogger(__name__).warning('Workflow lease rollback unavailable; closing connection',exc_info=True)
                finally:
                    try:lease.close()
                    except Exception:
                        logging.getLogger(__name__).warning('Workflow lease close failed',exc_info=True)
    thread=Thread(target=maintain_lease,name='a3-workflow-lease',daemon=True)
    previous=getattr(_workflow_state,'active',None)
    thread.start()
    try:
        if not ready.wait(15):raise TimeoutError('Workflow lease acquisition timed out')
        if state['error'] is not None:raise state['error']
        if state['acquired']:_workflow_state.active=state
        yield state['acquired']
    finally:
        _workflow_state.active=previous
        stop.set()
        thread.join(timeout=15)


def extraction_in_progress():
    with workflow_slot() as available:
        return not available


def dashboard_contracts(user):
    # Cards display one version and its latest status, never OCR/graph payloads.
    # Correlated ID filters keep one row per parent without loading history.
    from contracts.models import ContractVersion
    latest_run_id = LayoutExtractionRun.objects.filter(version_id=OuterRef('version_id')).order_by(
        '-started_at', '-pk',
    ).values('pk')[:1]
    runs = LayoutExtractionRun.objects.filter(pk=Subquery(latest_run_id)).only(
        'id', 'version_id', 'status', 'total_pages', 'total_blocks',
        'dense_index_status', 'colpali_index_status', 'started_at',
    ).order_by('-started_at', '-pk')
    latest_version_id = ContractVersion.objects.filter(contract_id=OuterRef('contract_id')).order_by(
        '-version_number', '-pk',
    ).values('pk')[:1]
    versions = ContractVersion.objects.filter(pk=Subquery(latest_version_id)).only('id', 'contract_id', 'version_number').order_by(
        '-version_number', '-pk',
    ).prefetch_related(Prefetch('layout_extraction_runs', queryset=runs))
    return list(visible_contracts(user).only('id', 'contract_code', 'title', 'updated_at').prefetch_related(
        Prefetch('versions', queryset=versions),
    ).order_by('-updated_at'))


def contract_for(user, contract_id, *, include_layout=True):
    rows=visible_contracts(user)
    if include_layout:
        rows=rows.prefetch_related('versions__files','versions__layout_extraction_runs__pages__blocks')
    return rows.get(pk=contract_id)


def latest_run(version, completed=False, *, status_only=False):
    if not version:
        return None
    if completed and not status_only:
        cached = getattr(version, '_prefetched_objects_cache', {}).get('layout_extraction_runs')
        if cached is not None:
            return next((run for run in cached if run.status == 'COMPLETED'), None)
    rows = version.layout_extraction_runs
    if status_only:
        # Polling must not download the full OCR/model traces or old page trees.
        from django.db.models.fields.json import KeyTransform,KeyTextTransform
        rows=rows.defer('structure').annotate(
            status_graph=KeyTransform('graph_index','structure'),
            status_processing=KeyTransform('processing','structure'),
            status_fingerprint=KeyTextTransform('source_fingerprint','structure'))
    return rows.filter(status='COMPLETED').first() if completed else rows.first()


def pages(run):
    return list(run.pages.all()) if run else []


def page_for(run, page_number):
    return LayoutExtractionPage.objects.prefetch_related('blocks').get(run=run, page_number=page_number)


def run_for_version(version, run_id):
    return version.layout_extraction_runs.get(pk=run_id)


def page_blocks(page):
    return list(page.blocks.all())


def page_numbers(run):
    return list(run.pages.order_by('page_number').values_list('page_number', flat=True))


def latest_file(version):
    return version.files.order_by('-uploaded_at').first()


def create_run(version, pipeline_name):
    return LayoutExtractionRun.objects.create(version=version, pipeline_name=pipeline_name, status='RUNNING')


def save_run(run, fields):
    ensure_write_connection()
    run.save(update_fields=fields)


def ensure_write_connection():
    """Reconnect an idle PostgreSQL connection before starting a write unit.

    OCR/model requests can outlive a pooled SSL connection. Never reconnect
    inside a transaction or replay a write whose commit result is unknown.
    """
    state=getattr(_workflow_state,'active',None)
    if state and state.get('error') is not None:
        raise RuntimeError('Workflow lease lost; refusing further writes') from state['error']
    from django.db import connections
    database=connections['default']
    if database.vendor!='postgresql' or database.in_atomic_block:return
    if database.connection is not None and not database.is_usable():
        database.close()
    database.ensure_connection()


def indexed_run(run):
    return LayoutExtractionRun.objects.select_related('version__contract').get(pk=run.pk)


def persist_rendered_page(run, page_number, image_path, width, height):
    """Persist the rendered page before Surya detects source blocks."""
    ensure_write_connection()
    return LayoutExtractionPage.objects.create(run=run, page_number=page_number,
        image_path=image_path, width=width, height=height)


@transaction.atomic
def persist_pages(run, result, *, incremental=False, cumulative=True):
    from ai_extract.models import LayoutExtractionPage, LayoutExtractionBlock
    from contracts.models import ContractContext
    contexts = []
    new_records = []
    total_blocks = 0
    for page_result in result.pages:
        defaults = dict(width=page_result.width, height=page_result.height,
                        image_path=page_result.image_path, extracted_text=page_result.text)
        if incremental:
            page, created = LayoutExtractionPage.objects.update_or_create(
                run=run, page_number=page_result.page_number, defaults=defaults)
            # Retrying a page replaces its derived blocks instead of appending
            # a second OCR copy. Other pages and historical runs are untouched.
            if not created:
                page.blocks.all().delete()
        else:
            page = LayoutExtractionPage.objects.create(run=run, page_number=page_result.page_number, **defaults)
        blocks = [LayoutExtractionBlock(page=page, block_order=block.order, layout_type=block.layout_type, text=block.text, bbox=block.bbox, structure_metadata=getattr(block, 'structure_metadata', {}), confidence=Decimal(str(round(max(0.0, min(1.0, block.confidence)), 4)))) for block in page_result.blocks]
        LayoutExtractionBlock.objects.bulk_create(blocks)
        new_records.extend(dict(chunk_id=f'block:{block.pk}', text=block.text,
            page_number=page.page_number, bbox=block.bbox, layout_type=block.layout_type,
            image_path=page.image_path, structure_metadata=block.structure_metadata) for block in blocks)
        total_blocks += len(blocks)
        contexts.append(ContractContext(version=run.version, context_type='raw_text', source=f'plan-a:page:{page_result.page_number}', content=page_result.text, relevance_score=Decimal('1.00') if page_result.text else Decimal('0.00')))
    contexts_query = ContractContext.objects.filter(version=run.version, context_type='raw_text',source__startswith='plan-a:page:')
    if incremental:
        contexts_query = contexts_query.filter(source__in=[c.source for c in contexts])
    contexts_query.delete()
    ContractContext.objects.bulk_create(contexts)
    if not cumulative:
        return new_records, total_blocks
    records = [dict(chunk_id=f'block:{block.pk}', text=block.text, page_number=page.page_number, bbox=block.bbox, layout_type=block.layout_type, image_path=page.image_path, structure_metadata=block.structure_metadata) for page in run.pages.prefetch_related('blocks').order_by('page_number') for block in page.blocks.all()]
    return records, total_blocks


def retire_old_page_contexts(run):
    """Remove obsolete page contexts only after a newer complete source exists."""
    from contracts.models import ContractContext
    if run.status!='COMPLETED' or LayoutExtractionRun.objects.filter(
            version_id=run.version_id,pk__gt=run.pk,status__in=['RUNNING','COMPLETED']).exists():
        return 0
    sources=[f'plan-a:page:{number}' for number in run.pages.values_list('page_number',flat=True)]
    deleted,_=ContractContext.objects.filter(version_id=run.version_id,context_type='raw_text',
        source__startswith='plan-a:page:').exclude(source__in=sources).delete()
    return deleted


def completed_runs(version_id=None):
    from django.db.models import Q
    rows = LayoutExtractionRun.objects.filter(status='COMPLETED').select_related('version__contract').order_by('-started_at','-pk')
    if version_id:
        rows = rows.filter(version_id=version_id)
    # Transaction-pooled Postgres connections cannot retain named cursors
    # across fetches. Keyset batches use ordinary queries and bounded memory.
    cursor=None
    while True:
        query=rows if cursor is None else rows.filter(
            Q(started_at__lt=cursor.started_at)|Q(started_at=cursor.started_at,pk__lt=cursor.pk))
        batch=list(query[:10])
        if not batch:return
        yield from batch
        cursor=batch[-1]


def structure_records(run):
    return [dict(chunk_id=f'block:{block.pk}', text=block.text, page_number=page.page_number,
        bbox=block.bbox, layout_type=block.layout_type, image_path=page.image_path, structure_metadata=block.structure_metadata)
        for page in run.pages.prefetch_related('blocks').order_by('page_number')
        for block in page.blocks.all()]


def graph_source_blocks(run, block_ids):
    """Resolve graph labels from original blocks of this extraction only."""
    if not block_ids:
        return {}
    rows = LayoutExtractionBlock.objects.filter(page__run_id=run.pk, pk__in=block_ids).values(
        'id', 'block_order', 'text', 'page__page_number')
    return {str(row['id']): row for row in rows}


def graph_reset_runs(contract):
    from django.db.models.fields.json import KeyTextTransform
    return LayoutExtractionRun.objects.filter(version__contract=contract).select_related('version__contract').defer(
        'structure').annotate(reset_source_fingerprint=KeyTextTransform('source_fingerprint','structure'))


def mark_graph_reset(contract):
    from ai_extract.models import RetrievalSnapshot
    RetrievalSnapshot.objects.filter(version__contract=contract).update(dirty=True,graph_scope={})
