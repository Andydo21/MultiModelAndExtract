"""Prepare pipeline screens and run the authorized A3 extraction workflow."""
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from types import SimpleNamespace
from django.db import transaction, close_old_connections
from django.utils import timezone
from .inference_runtime_service import PipelineProgress
from django.conf import settings
from ai_extract.services.contract_pipeline_service import LayoutExtractionService
from ai_extract.repositories import layout_repository
from NewIdea.repositories.contract_repository import contract_version

def image_url(path):
    return f"/{settings.MEDIA_URL.strip('/')}/{path.lstrip('/')}" if path else ''

def dashboard(user):
    cards = []
    for contract in layout_repository.dashboard_contracts(user):
        version = contract.latest_version
        cards.append(dict(contract=contract, version=version, run=layout_repository.latest_run(version)))
    return dict(cards=cards)

def pipeline(user, contract_id, version_id=None, graph_page=None, graph_relation='structure', graph_offset=0, graph_node=None):
    if version_id and not str(version_id).isdecimal():
        raise ValueError('Version ID không hợp lệ.')
    contract = layout_repository.contract_for(user, contract_id)
    version = contract_version(contract, version_id) if version_id else contract.latest_version
    run = layout_repository.latest_run(version)
    pages = layout_repository.pages(run)
    for page in pages:
        page.preview_url = image_url(page.image_path)
    from .graph_visualization_service import tree_payload
    from .graph_visualization_service import graph_payload
    return dict(contract=contract, version=version, run=run, pages=pages,
                can_extract=version is not None and version == contract.latest_version,
                graph_visualization=graph_payload(run, contract.pk, version.pk if version else None,
                                                  graph_page, graph_relation, graph_offset, graph_node),
                structure_visualization=tree_payload(run.structure or {}, contract.pk, version.pk) if run else {})

def inspect_page(user, contract_id, page_number, version_id=None, graph_relation='all', graph_offset=0, graph_node=None, run_id=None):
    if run_id and not str(run_id).isdecimal():
        raise ValueError('Invalid extraction run ID')
    if version_id and (not str(version_id).isdecimal()):
        raise ValueError('Version ID không hợp lệ.')
    contract = layout_repository.contract_for(user, contract_id)
    version = contract_version(contract, version_id)
    run = layout_repository.run_for_version(version, run_id) if run_id else layout_repository.latest_run(version, completed=True)
    page = layout_repository.page_for(run, page_number)
    page.preview_url = image_url(page.image_path)
    for block in layout_repository.page_blocks(page):
        if len(block.bbox) == 4:
            x1, y1, x2, y2 = block.bbox
            block.box_style = f'left:{x1 / 10:.2f}%;top:{y1 / 10:.2f}%;width:{max(0, x2 - x1) / 10:.2f}%;height:{max(0, y2 - y1) / 10:.2f}%;'
        else:
            block.box_style = 'display:none;'
    numbers = layout_repository.page_numbers(run)
    index = numbers.index(page_number)
    from .graph_visualization_service import page_payload
    from .graph_visualization_service import graph_payload
    return dict(contract=contract, version=version, run=run, page=page,
                graph_page_fixed=True,
                graph_visualization=graph_payload(run, contract.pk, version.pk, page_number, graph_relation, graph_offset, graph_node),
                page_visualization=page_payload(run, page), previous_page=numbers[index - 1] if index else None, next_page=numbers[index + 1] if index + 1 < len(numbers) else None)


def extract(user, contract_id, *, force_ocr=False, defer_colpali=True):
    contract = layout_repository.contract_for(user, contract_id)
    version = contract.latest_version
    if not version:
        raise ValueError('Contract has no version.')
    if type(force_ocr) is not bool or type(defer_colpali) is not bool:
        raise ValueError('Extraction options must be booleans')
    service=LayoutExtractionService()
    run = service.extract_version(version,force_ocr=force_ocr,defer_colpali=defer_colpali)
    processing=run.structure.get('processing', {})
    failed_pages = processing.get('failed_pages', [])
    incomplete = {key:processing.get(key,[]) for key in (
        'visual_ocr_incomplete_pages','body_ocr_incomplete_pages','entity_incomplete_pages','table_structure_incomplete_pages')}
    return dict(success=run.status == 'COMPLETED', partial=bool(failed_pages or any(incomplete.values())), error=run.error_message,
                pipeline_schema=run.structure.get('pregraph_analysis',{}).get('schema'),
                entity_schema=run.structure.get('knowledge_graph',{}).get('schema'),
                force_ocr=force_ocr,defer_colpali=defer_colpali,
                **incomplete,
                failed_pages=failed_pages, run_id=run.id, status=run.status, total_pages=run.total_pages, total_blocks=run.total_blocks, dense_index_status=run.dense_index_status, dense_points_count=run.dense_points_count, colpali_index_status=run.colpali_index_status, colpali_points_count=run.colpali_points_count, structure_status=run.structure.get('structure_status'), structure_analysis=run.structure.get('analysis', {}), graph_index=run.structure.get('graph_index', {}), knowledge_status=run.structure.get('knowledge_graph', {}).get('status', 'not_analyzed'))


def queue_identity(user, contract_id):
    contract=layout_repository.contract_for(user,contract_id)
    run=layout_repository.latest_run(contract.latest_version,completed=True)
    if not run or run.structure.get('graph_index',{}).get('status')!='ready':
        raise ValueError('A completed extraction with a ready graph is required')
    from .document_graph_service import validate_prepared
    from .gliner_identity_classifier import MODEL_ID,REVISION
    validate_prepared(run.structure)
    return dict(success=True,run_id=run.pk,identity_state='complete',model=MODEL_ID,revision=REVISION)


def reset_graph(user,contract_id):
    from django.core.exceptions import PermissionDenied
    from NewIdea.services.case_rules import can_review
    if not can_review(user):raise PermissionDenied('Only a document manager may reset its graph')
    contract=layout_repository.contract_for(user,contract_id)
    from .document_graph_service import graph_scope
    from ai_extract.repositories.document_graph_repository import reset_contract_graph
    with layout_repository.workflow_slot() as available:
        if not available:
            from .contract_pipeline_service import WorkflowBusyError
            raise WorkflowBusyError('A document workflow is active; graph reset was not performed')
        keys=[graph_scope(run,dict(source_fingerprint=run.reset_source_fingerprint))['key']
              for run in layout_repository.graph_reset_runs(contract) if run.reset_source_fingerprint]
        result=reset_contract_graph(contract.company_id,contract.pk,keys)
        layout_repository.mark_graph_reset(contract)
    return dict(success=True,contract_id=contract.pk,company_id=contract.company_id,**result)


def extraction_status(user, contract_id):
    contract=layout_repository.contract_for(user,contract_id,include_layout=False)
    run=layout_repository.latest_run(contract.latest_version,status_only=True)
    if not run:return dict(success=True,status='NOT_STARTED')
    identity=dict(state='not_ready')
    graph=run.status_graph or {}
    if run.status=='COMPLETED' and graph.get('status')=='ready':
        from .document_graph_service import graph_scope
        from ai_extract.repositories.document_graph_repository import identity_status
        identity=identity_status(graph_scope(run,dict(source_fingerprint=run.status_fingerprint)))
    return dict(success=True,run_id=run.pk,status=run.status,pages=run.total_pages,
        error=run.error_message,processing=run.status_processing or {},
        dense_status=run.dense_index_status,colpali_status=run.colpali_index_status,
        graph=graph,identity=identity)


def reindex_versions(version_id=None, *, dense_only=False,colpali_only=False):
    if dense_only and colpali_only:raise ValueError('Choose either dense-only or colpali-only')
    seen = set()
    for run in layout_repository.completed_runs(version_id):
        if run.version_id in seen:
            continue
        seen.add(run.version_id)
        if colpali_only:
            backfill_colpali(run)
            run.refresh_from_db()
        else:
            LayoutExtractionService()._index_plan_a_collections(run,dense_only=dense_only)
        yield run


def backfill_colpali(run,*,only_missing=True):
    """Embed stored figure images after extraction; preserve OCR, entities and dense vectors."""
    import hashlib
    from .contract_pipeline_service import PlanACollectionIndexer,IndexResult,WorkflowBusyError
    from .layout_extractor import visual_sources,colpali_point_id
    from .document_graph_service import sync_graph
    if type(only_missing) is not bool:raise ValueError('only_missing must be a boolean')
    with layout_repository.workflow_slot() as available:
        if not available:raise WorkflowBusyError('A document workflow is active; retry ColPali later')
        run=layout_repository.indexed_run(run)
        if run.status!='COMPLETED' or not run.structure.get('source_fingerprint'):
            raise ValueError('ColPali backfill requires a completed source extraction')
        if not getattr(settings,'COLPALI_ENABLED',True) or not PlanACollectionIndexer._colpali_url():
            raise ValueError('Enable/configure the ColPali endpoint before starting backfill; stored images are retained')
        previous_ids=set(run.structure.get('colpali_point_ids',[]))
        previous_count=run.colpali_points_count
        run.colpali_index_status='RUNNING';run.colpali_index_error=''
        layout_repository.save_run(run,['colpali_index_status','colpali_index_error'])
        visuals=[];indexer=None
        def embed():
            nonlocal visuals,indexer
            visuals=list(visual_sources(run))
            indexer=PlanACollectionIndexer()
            return indexer.index_colpali(run,visuals=visuals,skip_existing=only_missing)
        progress=PipelineProgress(run.pk)
        with progress.stage('colpali_backfill',only_missing=only_missing) as details:
            result=IndexResult.capture(embed)
            if result.status in {'FAILED','PARTIAL'} and not result.verified_inventory and (previous_count or previous_ids):
                ids=tuple(sorted(previous_ids|set(result.point_ids)))
                result=IndexResult('PARTIAL',max(len(ids),previous_count,result.indexed_count),result.error,ids)
            fields=result.apply_to(run,'colpali')
            acknowledged=set(result.point_ids);manifest=[]
            for visual in visuals:
                image_id=colpali_point_id(run.pk,visual.page.pk,visual.block.pk)
                path=Path(settings.MEDIA_ROOT)/visual.image_path
                manifest.append(dict(image_id=image_id,point_id=image_id,run_id=run.pk,
                    page_id=visual.page.pk,page_number=visual.page.page_number,block_id=visual.block.pk,
                    image_path=visual.image_path,image_sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
                    state='indexed' if image_id in acknowledged else 'pending'))
            run.structure['colpali_images']=manifest
            summary=dict(status=result.status,points=result.indexed_count,error=result.error,
                         only_missing=only_missing,images=len(manifest),graph=dict(status='not_updated'))
            run.structure['colpali_backfill']=summary
            # Persist acknowledged IDs before rebuilding HAS_EMBEDDING links.
            layout_repository.save_run(run,[*fields,'structure'])
            if acknowledged or previous_ids!=acknowledged:
                summary['graph']=sync_graph(run)
                if summary['graph']['status']=='ready':
                    run.structure['graph_index']=summary['graph']
                    layout_repository.save_run(run,['structure'])
                    try:
                        from NewIdea.services.knowledge_search_service import refresh_snapshot
                        refresh_snapshot(run.version)
                        summary['retrieval_snapshot']='refreshed'
                    except Exception as error:
                        summary['retrieval_snapshot']='unavailable:'+type(error).__name__
            if indexer is not None and result.status=='COMPLETED' and summary['graph']['status']=='ready':
                try:summary['cleanup']=indexer.prune_old_run_points(run,branches=('colpali',))
                except Exception as error:summary['cleanup']=dict(status='failed',error_type=type(error).__name__)
            layout_repository.save_run(run,['structure'])
            details.update(status=result.status,points=result.indexed_count,images=len(manifest))
            return dict(run_id=run.pk,**summary)


def enrich_entity_graph(run):
    """Enrich a completed source tree without redoing OCR or changing vector IDs."""
    from copy import deepcopy
    from .document_structure_service import attach_reading_streams
    from .document_graph_service import sync_graph
    from ai_extract.services.inference_runtime_service import PipelineProgress
    if run.status!='COMPLETED' or not run.structure.get('source_fingerprint'):
        raise ValueError('A completed source-grounded extraction is required')
    original=run.structure
    tree=deepcopy(original)
    for key in ('pregraph_analysis','reading_analysis','identity_resolution','structure_review'):
        tree.pop(key,None)
    tree=attach_reading_streams(tree)
    run.structure=tree
    progress=PipelineProgress(run.id)
    try:
        prepare_document(run,progress)
        with progress.stage('neo4j_entities') as details:
            graph=sync_graph(run)
            if graph['status']!='ready':raise RuntimeError('Neo4j publication unavailable: '+str(graph))
            tree['graph_index']=graph
            layout_repository.save_run(run,['structure'])
            details.update(status='ready',nodes=graph.get('nodes'),edges=graph.get('edges'))
    except Exception:
        run.structure=original
        raise
    return run


def rebuild_structures(version_id=None):
    from ai_extract.services.document_structure_service import analyze_document_structure
    count = 0
    for run in layout_repository.completed_runs(version_id):
        run.structure = analyze_document_structure(layout_repository.structure_records(run), run.version.contract.title)
        from .document_graph_service import enrich_and_sync
        enrich_and_sync(run)
        # Old dense points may contain different leaf IDs and heading vectors.
        # Retrieval must wait for reindexing rather than use stale points.
        run.dense_index_status = 'PENDING'
        run.colpali_index_status = 'PENDING'
        layout_repository.save_run(run, ['structure', 'dense_index_status', 'colpali_index_status'])
        count += 1
    return count


def refine_page_tables(page, progress, *, force=False):
    """Reconstruct PDF tables before source persistence, NER and indexing."""
    import base64
    import hashlib
    import io
    import json
    import math
    import requests
    from PIL import Image
    from ai_extract.repositories import page_inference_cache_repository as cache
    from .surya_inference_service import TABLE_CELL_POLICY
    from .document_structure_service import normalize_layout
    counts=dict(completed=0,unresolved=0)
    root=Path(settings.MEDIA_ROOT).resolve()
    endpoint=getattr(settings,'RAG_INFERENCE_URL','').rstrip('/')
    for block in getattr(page,'blocks',[]):
        metadata=getattr(block,'structure_metadata',{}) or {}
        # Native Office tables already supply their physical cell structure.
        if metadata.get('text_source')!='surya_ocr':
            continue
        if normalize_layout(getattr(block,'layout_type',''))!='table':
            # OCR may use HTML tables to arrange navigation, lists or page
            # furniture. Layout supplies the region type; markup cannot
            # promote these regions into document tables.
            if any(key in metadata for key in ('table_rows','table_cells','table_structure_status')):
                metadata=dict(metadata)
                for key in ('table_rows','table_cells','table_values','table_headers','table_structure_status',
                            'table_header_origin','table_structure_review','table_structure_policy','table_cell_bbox_scope'):
                    metadata.pop(key,None)
                metadata['table_markup_status']='ignored_non_table_layout'
                block.structure_metadata=metadata
            continue
        with progress.stage('table_cells',page=page.page_number) as details:
            try:
                path=(root/(metadata.get('structure_image_path') or page.image_path)).resolve()
                if not path.is_relative_to(root):raise ValueError('Table image outside media root')
                bbox=block.bbox
                if len(bbox)!=4 or any(not math.isfinite(float(v)) for v in bbox):
                    raise ValueError('Invalid table source bbox')
                with Image.open(path) as image:
                    width,height=image.size;pad=round(width*.012)
                    box=(max(0,int(width*bbox[0]/1000)-pad),max(0,int(height*bbox[1]/1000)-pad),
                         min(width,int(width*bbox[2]/1000)+pad),min(height,int(height*bbox[3]/1000)+pad))
                    if box[2]<=box[0] or box[3]<=box[1]:raise ValueError('Empty table crop')
                    crop=image.crop(box).convert('RGB');buffer=io.BytesIO();crop.save(buffer,format='PNG')
                blob=buffer.getvalue();digest=hashlib.sha256(blob).hexdigest()
                source_cells=metadata.get('table_cells',[])
                key=hashlib.sha256(json.dumps([TABLE_CELL_POLICY,endpoint,digest,source_cells],
                    sort_keys=True,ensure_ascii=False).encode()).hexdigest()
                stored=None if force else cache.read(key)
                result=stored.get('table_result') if stored else None
                if result is None:
                    if not endpoint:raise ValueError('Surya table endpoint unavailable')
                    response=requests.post(endpoint+'/rag/table_cells',json=dict(
                        image_base64=base64.b64encode(blob).decode(),source_cells=source_cells),timeout=(5,1800))
                    if response.status_code==422:
                        raise ValueError(str(response.json().get('detail','Physical table unresolved'))[:300])
                    response.raise_for_status();result=response.json()
                validate_table_cells(result,crop.size)
                proof=result.get('table_structure_review',{})
                if (result.get('table_structure_policy')!=TABLE_CELL_POLICY
                        or proof.get('image_size')!=list(crop.size)):
                    raise ValueError('Table inference policy or image size mismatch')
                if not stored:
                    cache.write(key,dict(model='surya-ocr-v2',layout=[],ocr_lines=[],table_result=result))
                review=dict(result['table_structure_review'],source_image_sha256=digest,
                    page_image_path=path.relative_to(root).as_posix(),crop_bbox_pixels=list(box),
                    page_image_size=[width,height],cache_hit=bool(stored))
                metadata=dict(metadata,table_original_text=block.text,**result)
                metadata['table_structure_review']=review
                block.text='\n'.join('\t'.join(row) for row in result['table_rows'])
                metadata['text_source']='surya_table_cell_ocr'
                # Cell boxes exported to the graph use the same normalized
                # page coordinates as the original layout blocks.
                for cell in metadata['table_cells']:
                    x0,y0,x1,y1=cell['bbox']
                    cell['bbox']=[(box[0]+x0)*1000/width,(box[1]+y0)*1000/height,
                                  (box[0]+x1)*1000/width,(box[1]+y1)*1000/height]
                metadata['table_cell_bbox_scope']='page_normalized_1000'
                metadata.pop('table_values',None)
                metadata['table_source_text_retained']=True
                counts['completed']+=1
                details.update(status='cell_aware',rows=review['rows'],columns=review['columns'],cache_hit=bool(stored))
            except (OSError,ValueError,KeyError,TypeError,requests.RequestException) as error:
                # Keep page OCR for diagnostics, but do not substitute it for
                # the requested table-specific cell reading.
                metadata=dict(metadata,table_original_text=block.text,table_structure_status='unresolved',table_structure_policy=TABLE_CELL_POLICY,
                    table_structure_review=dict(status='unresolved',error_type=type(error).__name__,reason=str(error)[:300]))
                metadata.pop('table_rows',None);metadata.pop('table_cells',None);metadata.pop('table_values',None)
                block.text=''
                counts['unresolved']+=1
                details.update(status='unresolved',error_type=type(error).__name__)
            block.structure_metadata=metadata
    if any(counts.values()):page.text='\n\n'.join(block.text for block in page.blocks)
    return counts


def validate_table_cells(result, image_size):
    """Reject holes, duplicate anchors, overlap and coordinates outside the crop."""
    import math
    rows=result['table_rows'];cells=result['table_cells'];review=result['table_structure_review']
    if result.get('table_structure_status')!='cell_aware' or not rows or not rows[0]:
        raise ValueError('Incomplete physical table')
    height=len(rows);width=len(rows[0])
    if (height>200 or width>64 or height*width>2048 or len(cells)>2048 or review['rows']!=height or review['columns']!=width
            or any(len(row)!=width or any(not isinstance(text,str) for text in row) for row in rows)):
        raise ValueError('Invalid physical table dimensions')
    occupied=set()
    for cell in cells:
        r,c,rs,cs=(cell[k] for k in ('row','column','rowspan','colspan'))
        if any(type(v) is not int or v<1 for v in (r,c,rs,cs)) or r+rs-1>height or c+cs-1>width:
            raise ValueError('Invalid physical table span')
        if rows[r-1][c-1]!=cell.get('text'):raise ValueError('Cell text is not aligned to grid')
        bounds=cell['bbox']
        if (len(bounds)!=4 or any(not math.isfinite(float(v)) for v in bounds)
                or not 0<=bounds[0]<bounds[2]<=image_size[0]
                or not 0<=bounds[1]<bounds[3]<=image_size[1]):raise ValueError('Invalid physical cell bbox')
        slots={(a,b) for a in range(r,r+rs) for b in range(c,c+cs)}
        if occupied & slots:raise ValueError('Duplicate or overlapping physical cells')
        occupied.update(slots)
    if len(occupied)!=height*width:raise ValueError('Physical grid contains holes')


class PageExtractionSession:
    def __init__(self, run, *, force_ocr=False, defer_colpali=False):
        self.run = run
        self.force_ocr=force_ocr
        self.defer_colpali=defer_colpali
        self.page_states = []
        self.colpali_results = {}
        self.visual_tasks = {}
        self.executor = None
        self.progress = PipelineProgress(run.pk)
        self.expected_pages = None
        self.records = []
        self.failed_pages = []
        self.visual_inputs = {}
        self.completed_pages = set()

    def __enter__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='a3-colpali-page')
        return self

    def __exit__(self, *exc):
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)

    def _submit_visual(self, page_number):
        if self.executor is None:
            raise RuntimeError('PageExtractionSession must be used as a context manager')
        if self.defer_colpali or not getattr(settings,'COLPALI_ENABLED',True):
            from .contract_pipeline_service import IndexResult
            task=Future();task.set_result(IndexResult('SKIPPED'))
            self.visual_tasks[page_number]=task
            return
        self.visual_tasks[page_number] = self.executor.submit(self._index_visual, page_number)

    def _index_visual(self, page_number):
        from .contract_pipeline_service import PlanACollectionIndexer, IndexResult
        # ORM connections belong to their thread. Only the main thread saves
        # run status; the background branch writes its own Qdrant points.
        close_old_connections()
        try:
            with self.progress.stage('colpali', page=page_number, total_pages=self.expected_pages) as details:
                def index():
                    if self.defer_colpali or not getattr(settings,'COLPALI_ENABLED',True):
                        return IndexResult('SKIPPED')
                    prepared = self.visual_inputs.get(page_number)
                    if prepared is not None:
                        run, visuals = prepared
                    else:
                        from .layout_extractor import visual_sources
                        with self.progress.stage('colpali_sources', page=page_number):
                            run = layout_repository.indexed_run(self.run)
                            visuals = list(visual_sources(run, page_number=page_number))
                    return PlanACollectionIndexer().index_colpali(run, page_number=page_number, visuals=visuals)
                result = IndexResult.capture(index)
                details.update(status=result.status, points=result.indexed_count)
                return result
        finally:
            close_old_connections()

    def start_page_image(self, page_number, image_path, width, height, *, total_pages=None):
        if page_number in self.completed_pages or any(s['page_number']==page_number for s in self.page_states):
            raise ValueError('Duplicate extraction page')
        if self.visual_tasks or any(s.get('structure') == 'RUNNING' for s in self.page_states):
            raise RuntimeError('Finish both branches of the previous page first')
        relative = Path(image_path).resolve().relative_to(Path(settings.MEDIA_ROOT).resolve()).as_posix()
        layout_repository.persist_rendered_page(self.run, page_number, relative, width, height)
        self.expected_pages = total_pages or self.expected_pages
        self.progress.emit('render', 'completed', page=page_number, total_pages=self.expected_pages,
                           width=width, height=height)
        self.page_states.append(dict(page_number=page_number, structure='RUNNING',
                                     graph='PENDING', dense='DEFERRED', colpali='PENDING'))
        self.run.colpali_index_status = 'RUNNING'
        self._save_progress('pages')


    def fail_page(self, page_number, error):
        from .contract_pipeline_service import IndexResult
        self.failed_pages.append(dict(page_number=page_number, stage='surya', error_type=type(error).__name__))
        state = next(s for s in self.page_states if s['page_number'] == page_number)
        task = self.visual_tasks.pop(page_number, None)
        result = task.result() if task else IndexResult('SKIPPED')
        self._record_visual_result(page_number, result)
        state.update(structure='FAILED', graph='SKIPPED', error_type=type(error).__name__)
        self._save_progress('pages')
        self.progress.emit('page', 'failed_continuing', page=page_number,
                           total_pages=self.expected_pages, error_type=type(error).__name__)

    def process_page(self, page):
        from ai_extract.services.document_structure_service import analyze_document_structure
        if page.page_number in self.completed_pages:
            raise ValueError('Duplicate extraction page')
        run = self.run
        table_counts=refine_page_tables(page,self.progress,force=self.force_ocr)
        layout_repository.ensure_write_connection()
        with transaction.atomic():
            records, _ = layout_repository.persist_pages(run, SimpleNamespace(pages=[page]),
                                                         incremental=True, cumulative=False)
        self.records.extend(records)
        self.colpali_results.pop(page.page_number,None)
        state = next((s for s in self.page_states if s['page_number'] == page.page_number), None)
        if state is None:
            # Office sources do not have a full rendered page; their embedded
            # visuals become available only after parser persistence.
            state = dict(page_number=page.page_number, colpali='RUNNING', dense='DEFERRED')
            self.page_states.append(state)
        # Surya blocks now exist: only their confirmed figures enter ColPali.
        self._submit_visual(page.page_number)
        with self.progress.stage('structure', page=page.page_number, total_pages=self.expected_pages) as details:
            run.structure = analyze_document_structure(self.records, run.version.contract.title)
            run.structure['colpali_point_ids'] = [pid for r in self.colpali_results.values() for pid in r.point_ids]
            details.update(status=run.structure['structure_status'], chunks=len(run.structure['leaves']))
        state.update(structure=run.structure['structure_status'],
                     entities='DEFERRED_UNTIL_ALL_PAGES',
                     graph='DEFERRED_UNTIL_ANALYSIS', dense='DEFERRED')
        state['table_cells']=table_counts
        visual_states = [(record.get('structure_metadata') or {}).get('visual_ocr_status')
                         for record in records if (record.get('structure_metadata') or {}).get('visual_ocr_status')]
        state['visual_ocr'] = dict(regions=len(visual_states),failed=visual_states.count('failed'),
                                  unreadable=sum(v in {'unreadable','partial','unresolved'} for v in visual_states))
        state['body_ocr_failed_regions']=sum(
            not (record.get('structure_metadata') or {}).get('visual_depth')
            and (record.get('structure_metadata') or {}).get('ocr_status') in {'failed','rejected','unresolved'}
            for record in records)
        run.total_pages += 1
        run.total_blocks += len(page.blocks)
        run.colpali_index_status = 'RUNNING'
        self._save_progress('pages')
        with self.progress.stage('wait_colpali', page=page.page_number):
            result = self.visual_tasks.pop(page.page_number).result()
        self._record_visual_result(page.page_number, result)
        run.structure['graph_index']=dict(status='analysis_pending',
            reason='Whole-document reading and entity identity must finish before Neo4j publication')
        self._save_progress('pages')
        self.progress.emit('page', 'completed', page=page.page_number, total_pages=self.expected_pages,
                           blocks=len(page.blocks), graph=state['graph'], colpali=result.status, dense='DEFERRED')
        self.completed_pages.add(page.page_number)

    def _record_visual_result(self, page_number, result):
        from .contract_pipeline_service import IndexResult
        previous = self.colpali_results.get(page_number)
        if previous and result.status != 'COMPLETED':
            # Upserts reuse deterministic IDs. A failed retry must neither lose
            # acknowledged vectors nor count the same point twice.
            ids = tuple(sorted(set(previous.point_ids) | set(result.point_ids)))
            tracked = (len(previous.point_ids) == previous.indexed_count and
                       len(result.point_ids) == result.indexed_count)
            count = len(ids) if tracked else max(previous.indexed_count, result.indexed_count)
            result = IndexResult('PARTIAL' if count else 'FAILED', count,
                                 result.error or previous.error, ids)
        self.colpali_results[page_number] = result
        state = next(s for s in self.page_states if s['page_number'] == page_number)
        state.update(colpali=result.status, colpali_points=result.indexed_count,
                     colpali_error=result.error, colpali_attempts=state.get('colpali_attempts', 0) + 1)
        IndexResult.combine(self.colpali_results.values()).apply_to(self.run, 'colpali')
        return result

    def _retry_failed_visuals(self):
        pending = [page for page, result in self.colpali_results.items()
                   if result.status in {'FAILED', 'PARTIAL'}]
        if not pending:
            return
        self.run.colpali_index_status = 'RUNNING'
        self._save_progress('retrying_colpali')
        self.progress.emit('colpali_retry', 'started', pages=pending, max_attempts=1)
        for page_number in pending:
            with self.progress.stage('colpali_retry_page', page=page_number, attempt=2) as details:
                self._submit_visual(page_number)
                result = self._record_visual_result(page_number, self.visual_tasks.pop(page_number).result())
                details.update(status=result.status, points=result.indexed_count)
            self._save_progress('retrying_colpali')
        remaining = [page for page, result in self.colpali_results.items()
                     if result.status in {'FAILED', 'PARTIAL'}]
        self.progress.emit('colpali_retry', 'completed', pages=pending, missing_pages=remaining)

    def _save_progress(self, phase):
        self.run.structure['processing'] = dict(phase=phase, pages=self.page_states,
                                               force_ocr=self.force_ocr,defer_colpali=self.defer_colpali,
                                               colpali_policy='deferred_backfill' if self.defer_colpali or not getattr(settings,'COLPALI_ENABLED',True) else 'inline_opt_in',
                                               pages_total=self.expected_pages,
                                               dense_policy='after_complete_structure',
                                               failed_pages=list(self.failed_pages),
                                               table_structure_incomplete_pages=[s['page_number'] for s in self.page_states
                                                   if s.get('table_cells',{}).get('unresolved')],
                                               colpali_missing_pages=[page for page, result in self.colpali_results.items()
                                                                      if result.status in {'FAILED', 'PARTIAL'}],
                                               global_relations='PENDING' if phase == 'pages' else phase)
        layout_repository.save_run(self.run, ['structure', 'total_pages', 'total_blocks',
            'colpali_index_status', 'colpali_points_count', 'colpali_index_error',
            'dense_index_status', 'dense_points_count', 'dense_index_error'])

    def finish(self):
        from .contract_pipeline_service import PlanACollectionIndexer, IndexResult
        from .document_graph_service import sync_graph
        run = self.run
        if self.visual_tasks:
            raise RuntimeError('Cannot finalize while a page is still processing')
        if not self.records and self.failed_pages:
            self._retry_failed_visuals()
            run.status = 'FAILED'
            run.error_message = 'OCR failed on every page; rendered pages and visual index retained.'
            run.completed_at = timezone.now()
            self._save_progress('finished')
            layout_repository.save_run(run, ['status', 'error_message', 'completed_at'])
            self.progress.emit('pipeline', 'partial', failed_pages=self.failed_pages, pages=0)
            return
        if self.failed_pages:
            IndexResult.combine(self.colpali_results.values()).apply_to(run,'colpali')
            run.structure['graph_index']=dict(status='incomplete_ocr')
            run.status='FAILED'
            run.error_message='OCR incomplete; final graph was not published. Failed pages: '+','.join(
                str(p['page_number']) for p in self.failed_pages)
            run.completed_at=timezone.now()
            self._save_progress('incomplete_ocr_no_graph')
            layout_repository.save_run(run,['status','error_message','completed_at'])
            self.progress.emit('pipeline','partial',failed_pages=self.failed_pages,graph='not_published')
            return
        self._save_progress('finalizing')
        # Only the whole-document pass extracts entities and relations.
        run.structure['knowledge_graph'] = dict(status='disabled',provider='source_layout',
            entities=[],relations=[],source_relations=[],errors=[])
        run.structure['visual_graph'] = dict(status='disabled', images=[], objects=[], relations=[], text_links=[])
        if getattr(settings,'A3_KNOWLEDGE_ENABLED',False):
            if getattr(settings,'A3_ENTITY_PROVIDER','disabled')!='gliner':
                raise RuntimeError('The current whole-document pipeline requires the configured GLiNER provider')
            prepare_document(run,self.progress,defer_related_passages=True)
            for state in self.page_states:
                state['entities']=run.structure['knowledge_graph']['status']
            self._save_progress('analysis_complete_before_graph')
        # Dense text uses the completed source hierarchy.
        run.dense_index_status = 'RUNNING'
        self._save_progress('indexing_dense')
        with self.progress.stage('dense_index', pages=run.total_pages,
                                 chunks=len(run.structure.get('leaves', []))) as details:
            dense = IndexResult.capture(lambda: PlanACollectionIndexer().index_dense(run, replace_existing=False))
            dense.apply_to(run, 'dense')
            details.update(status=dense.status, points=dense.indexed_count)
        if getattr(settings,'A3_RELATED_PASSAGES_ENABLED',True) and 'related_passages' not in run.structure:
            prepare_related_passages(run,self.progress)
        self._retry_failed_visuals()
        IndexResult.combine(self.colpali_results.values()).apply_to(run, 'colpali')
        with self.progress.stage('neo4j_final', pages=run.total_pages, chunks=len(run.structure.get('leaves', []))) as details:
            run.structure['graph_index'] = sync_graph(run)
            details['structure_review']=run.structure.get('structure_review',{}).get('counts',{})
            details.update(status=run.structure['graph_index']['status'])
        self.finalize_graph()

    def finalize_graph(self):
        """Finish a prepared run, including one resumed after graph publication failure."""
        run = self.run
        if getattr(settings,'A3_KNOWLEDGE_ENABLED',False) and run.structure['graph_index']['status']!='ready':
            run.status = 'FAILED'
            run.error_message = 'Prepared graph could not be published: '+str(run.structure['graph_index'])
            run.completed_at = timezone.now()
            self._save_progress('graph_publication_failed')
            layout_repository.save_run(run, ['status', 'error_message', 'completed_at'])
            self.progress.emit('pipeline', 'graph_publication_failed', graph=run.structure['graph_index'])
            return
        self._save_progress('finished')
        visual_missing = [s['page_number'] for s in self.page_states if
                          s.get('visual_ocr',{}).get('failed') or s.get('visual_ocr',{}).get('unreadable')]
        run.structure['processing']['visual_ocr_incomplete_pages'] = visual_missing
        body_missing=[s['page_number'] for s in self.page_states if s.get('body_ocr_failed_regions')]
        run.structure['processing']['body_ocr_incomplete_pages']=body_missing
        entity_missing=[s['page_number'] for s in self.page_states if s.get('entities') in {'partial','unavailable'}]
        run.structure['processing']['entity_incomplete_pages']=entity_missing
        run.structure['processing']['global_relations'] = dict(
            graph=run.structure.get('graph_index', {}).get('status'),
            knowledge=run.structure.get('knowledge_graph', {}).get('status'),
            visual=run.structure.get('visual_graph', {}).get('status'))
        run.structure['processing']['global_relations']['document_relations']=run.structure.get('knowledge_graph',{}).get('document_relations',{})
        related=run.structure.get('related_passages',{})
        run.structure['processing']['global_relations']['related_passages']={k:v for k,v in related.items() if k!='links'}
        related_missing=related.get('status') in {'unavailable','partial'}
        review=run.structure.get('structure_review',{})
        run.structure['processing']['global_relations']['structure_review']={k:review[k] for k in ('status','counts','relations_total','errors') if k in review}
        run.status = 'FAILED' if self.failed_pages else 'COMPLETED'
        if self.failed_pages:
            run.error_message = 'OCR incomplete on pages: '+', '.join(str(p['page_number']) for p in self.failed_pages)+'. Successful pages retained; rerun reuses completed OCR cache.'
        elif visual_missing:
            run.error_message = 'Visual-region OCR incomplete on pages: '+', '.join(map(str,visual_missing))+'. Readable text retained; failed requests retry on rerun, unreadable crops remain marked.'
        if body_missing:
            run.error_message=(getattr(run,'error_message','')+'\nBody OCR incomplete on pages: '+', '.join(map(str,body_missing))+'. Failed source regions are excluded from retrieval and entities.').strip()
        if entity_missing:
            run.error_message=(getattr(run,'error_message','')+'\nEntity extraction incomplete on pages: '+', '.join(map(str,entity_missing))+'. Source text, structure and dense index retained.').strip()
        if related_missing:
            run.error_message=(getattr(run,'error_message','')+'\nRelated-passage retrieval unavailable; no similarity links were invented.').strip()
        document_missing=run.structure.get('knowledge_graph',{}).get('document_relations',{}).get('status') in {'partial','unavailable'}
        if document_missing:
            run.error_message=(getattr(run,'error_message','')+'\nDocument relation analysis incomplete; source structure and local relation proposals retained.').strip()
        review_missing=review.get('status')=='partial'
        if review_missing:
            run.error_message=(getattr(run,'error_message','')+'\nStructure review incomplete; source proposals and pending review status retained.').strip()
        table_missing=run.structure['processing'].get('table_structure_incomplete_pages',[])
        if table_missing:
            run.error_message=(getattr(run,'error_message','')+'\nPhysical table structure unresolved on pages: '
                +', '.join(map(str,table_missing))+'. Source text retained; candidate rows/cells were not published.').strip()
        graph_ready = run.structure['graph_index']['status'] == 'ready'
        if not graph_ready:
            run.error_message=(getattr(run,'error_message','')+'\nNeo4j index unavailable; extracted sources and vector indexes retained.').strip()
        run.completed_at = timezone.now()
        layout_repository.save_run(run, ['structure', 'status', 'error_message', 'completed_at'])
        from .document_graph_service import cleanup_old_run_graphs
        with self.progress.stage('neo4j_cleanup') as details:
            details.update(cleanup_old_run_graphs(run))
        with self.progress.stage('artifact_cleanup') as details:
            details.update(retire_superseded_artifacts(run))
        if run.status == 'COMPLETED':
            with self.progress.stage('retrieval_index'):
                from NewIdea.services.knowledge_search_service import refresh_snapshot
                refresh_snapshot(run.version)
        indexes_ready = graph_ready and run.dense_index_status == 'COMPLETED' and run.colpali_index_status in {'COMPLETED', 'SKIPPED'}
        self.progress.emit('pipeline', 'partial' if self.failed_pages else 'completed_with_source_warnings' if visual_missing or body_missing or entity_missing or document_missing or review_missing or related_missing or table_missing else 'completed' if indexes_ready else 'completed_with_index_errors',
                           pages=run.total_pages, blocks=run.total_blocks, indexes_ready=indexes_ready,
                           dense=run.dense_index_status, colpali=run.colpali_index_status,
                           graph=run.structure['graph_index']['status'])


def retire_superseded_artifacts(run):
    """Retire derived indexes through the pipeline, preserving original run history."""
    from .contract_pipeline_service import PlanACollectionIndexer
    if getattr(run,'status',None)!='COMPLETED':return dict(status='skipped')
    result={}
    try:
        result['page_contexts_deleted']=layout_repository.retire_old_page_contexts(run)
        result['vectors']=PlanACollectionIndexer().prune_old_run_points(run)
        result['status']='completed' if all(v in {'completed','skipped','not_created'} for v in result['vectors'].values()) else 'partial'
    except Exception as error:
        result.update(status='failed',error_type=type(error).__name__)
    run.structure.setdefault('processing',{})['artifact_cleanup']=result
    layout_repository.save_run(run,['structure'])
    return result


def prepare_related_passages(run,progress):
    from .document_graph_service import infer_related_passages
    with progress.stage('related_passages',pages=run.total_pages,provider='bge') as details:
        try:
            run.structure['related_passages']=infer_related_passages(run.structure,
                candidate_limit=getattr(settings,'A3_RELATED_PASSAGE_CANDIDATES',5),
                link_limit=getattr(settings,'A3_RELATED_PASSAGE_LIMIT',2),
                min_logit=getattr(settings,'A3_RELATED_PASSAGE_MIN_LOGIT',0.))
        except Exception as error:
            run.structure['related_passages']=dict(status='unavailable',links=[],error_type=type(error).__name__,
                semantics='content_similarity_navigation_only')
        details.update(status=run.structure['related_passages']['status'],
                       links=len(run.structure['related_passages']['links']))
    proof=run.structure.get('pregraph_analysis')
    if proof is not None and 'related_passages' not in proof['stages']:
        proof['stages'].append('related_passages')
    layout_repository.save_run(run,['structure'])


def prepare_document(run,progress,*,defer_related_passages=False):
    from .document_structure_service import prepare_reading
    from .gliner_identity_classifier import resolve_identities
    from .gliner_entity_service import extract_graph_entities
    from .document_graph_service import PREGRAPH_PIPELINE_SCHEMA
    tree=run.structure
    if not tree.get('source_fingerprint'):
        raise ValueError('Original whole-document source fingerprint is required')
    tree['graph_index']=dict(status='analysis_pending')
    tree['entity_index']=dict(status='DISABLED',points=0,reason='entity_embeddings_removed')
    tree.pop('pregraph_analysis',None)
    tree.pop('identity_resolution',None)
    tree.pop('knowledge_graph',None)
    tree.pop('structure_review',None)
    tree.pop('related_passages',None)
    with progress.stage('pregraph_source_reading',pages=run.total_pages) as details:
        prepare_reading(run)
        if tree['reading_analysis']['status']!='complete':
            raise RuntimeError('Surya source reading preparation incomplete before graph publication')
        details.update(status=tree['reading_analysis']['status'],
            sources=len(tree['reading_analysis']['source_ids']),
            source_order_preserved=True,new_reading_edges=0,
            provider='surya_source',semantic_review='disabled')
        layout_repository.save_run(run,['structure'])
    with progress.stage('whole_document_entities',pages=run.total_pages,provider='gliner') as details:
        knowledge=extract_graph_entities(tree)
        if knowledge['status'] not in {'complete','empty'}:
            raise RuntimeError('Whole-document entity extraction incomplete: '+str(knowledge['errors']))
        tree['knowledge_graph']=knowledge
        details.update(status=knowledge['status'],entities=len(knowledge['entities']),
            entity_relations=len(knowledge['relations']),
            sources_completed=len(knowledge['source_coverage']['completed_ids']))
        layout_repository.save_run(run,['structure'])
    knowledge['document_relations']=dict(status='disabled',provider=None,
        reason='document_reference_resolution_removed',accepted_reference_relations=0)
    with progress.stage('pregraph_identity',pages=run.total_pages,model='fastino/gliner2-base-v1') as details:
        result=resolve_identities(run,tree)
        details.update(status=result['status'],mentions=result['mentions'],
            linked=result['linked'],unresolved=result['unresolved'],canonical_entities=result['canonical_entities'])
        layout_repository.save_run(run,['structure'])
    if getattr(settings,'A3_RELATED_PASSAGES_ENABLED',True) and not defer_related_passages:
        prepare_related_passages(run,progress)
    elif not getattr(settings,'A3_RELATED_PASSAGES_ENABLED',True):
        tree['related_passages']=dict(status='disabled',links=[])
    tree['pregraph_analysis']=dict(schema=PREGRAPH_PIPELINE_SCHEMA,status='complete',
        source_fingerprint=tree['source_fingerprint'],
        stages=['surya_source_reading','whole_document_entities','same_name_context_identity']+
            (['related_passages'] if getattr(settings,'A3_RELATED_PASSAGES_ENABLED',True) and not defer_related_passages else []),
        disabled_stages=['document_context_relations'],
        graph_used_as_analysis_input=False)
    return tree

