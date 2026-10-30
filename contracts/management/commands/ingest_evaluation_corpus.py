"""Import real selected PDFs through ContractService and the production A3 service."""
import contextlib
import hashlib
import json
import logging
import sys
import time
from datetime import timedelta
from pathlib import Path

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.files import File
from django.core.management.base import BaseCommand, CommandError
from django.db import close_old_connections, transaction
from django.db import OperationalError
from django.utils import timezone
from evaluation.batch_io import save_state, Tee


def db_read(operation,attempts=4):
    for attempt in range(attempts):
        try:
            return operation()
        except OperationalError:
            close_old_connections()
            if attempt+1==attempts:
                raise
            time.sleep(min(2**attempt,8))


def file_sha(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):
            result.update(block)
    return result.hexdigest()


def run_state(version):
    from ai_extract.models import LayoutExtractionRun
    run=db_read(lambda:LayoutExtractionRun.objects.filter(version=version).order_by('-id').first())
    if not run:
        return dict(status='NOT_STARTED',run_id=None)
    return dict(run_id=run.pk,status=run.status,pages=run.total_pages,blocks=run.total_blocks,
        dense_status=run.dense_index_status,dense_points=run.dense_points_count,
        dense_error=run.dense_index_error,colpali_status=run.colpali_index_status,
        colpali_points=run.colpali_points_count,colpali_error=run.colpali_index_error,
        graph=run.structure.get('graph_index',{}),processing=run.structure.get('processing',{}),
        error=run.error_message,started_at=run.started_at.isoformat() if run.started_at else None)


class Command(BaseCommand):
    help='Register real evaluation PDFs and run the normal A3 OCR/graph/vector pipeline sequentially.'
    requires_system_checks=[]
    requires_migrations_checks=False

    def add_arguments(self,parser):
        parser.add_argument('--username',required=True)
        parser.add_argument('--selection',default='evaluation/benchmarks/corpus20_selection.json')
        parser.add_argument('--output',default='evaluation/benchmarks/corpus20/a3_ingestion.json')
        parser.add_argument('--log',default='evaluation/benchmarks/corpus20/a3_ingestion.log')
        parser.add_argument('--status',action='store_true')
        parser.add_argument('--preflight',action='store_true')
        parser.add_argument('--register-only',action='store_true')
        parser.add_argument('--force-reextract',action='store_true',
                            help='Create new A3 runs for completed PDFs; retain the previous run IDs in this batch report.')
        parser.add_argument('--defer-colpali',action='store_true',help='Explicitly defer visual indexing while its remote service is unavailable.')

    def handle(self,*args,**options):
        from contracts.models import Contract, ContractVersion
        from contracts.services.contract_service import ContractService
        from contracts.services.document_encryption_service import decrypt_pdf
        from ai_extract.services.contract_pipeline_service import LayoutExtractionService
        root=Path(settings.BASE_DIR)
        if options['defer_colpali']:
            settings.COLPALI_ENABLED=False
        user=get_user_model().objects.select_related('company','role').get(username=options['username'],is_active=True)
        if not user.is_superuser and (not user.role or user.role.role_name.upper() not in ('MANAGER','ADMIN')):
            raise CommandError('A manager/admin account is required for document import')
        if user.company_id is None:
            raise CommandError('Select an existing account with a company; no company is fabricated')
        if options['preflight']:
            self.check_services(require_colpali=not options['defer_colpali'])
            return
        selection_config=json.loads((root/options['selection']).read_text(encoding='utf-8'))
        selection=selection_config['documents']
        source_root=(root/'documents').resolve()
        sources=[]
        for alias,name in selection.items():
            path=(root/name).resolve()
            if not path.is_relative_to(source_root) or path.suffix.lower()!='.pdf' or not path.is_file():
                raise CommandError('Selected input must be an actual PDF in documents/: '+name)
            if selection_config.get('max_document_pages') is not None:
                import pymupdf
                with pymupdf.open(path) as pdf:
                    if len(pdf)>selection_config['max_document_pages']:
                        raise CommandError('PDF exceeds selected maximum page count: '+name)
            sources.append(dict(alias=alias,path=path,sha256=file_sha(path),source_path=name))
        if len({s['sha256'] for s in sources})!=len(sources):
            raise CommandError('Selection contains byte-identical documents')
        output=root/options['output']
        output.parent.mkdir(parents=True,exist_ok=True)
        if options['status']:
            if not output.exists():
                self.stdout.write('No ingestion manifest yet')
                return
            report=json.loads(output.read_text(encoding='utf-8'))
            for entry in report['documents']:
                if entry.get('version_id'):
                    version=ContractVersion.objects.get(pk=entry['version_id'],contract__company_id=user.company_id)
                    entry['a3']=run_state(version)
            self.stdout.write(json.dumps(report,ensure_ascii=False,indent=2))
            return
        self.check_services(require_models=not options['register_only'],require_colpali=not options['defer_colpali'])
        if options['defer_colpali']:
            self.stdout.write('ColPali explicitly deferred: remote endpoint unavailable; visual indexing is not completed.')
        report=dict(selection=options['selection'],username=user.username,company_id=user.company_id,
            started_at=timezone.now().isoformat(),status='REGISTERING',documents=[],
            log_path=options['log'],state_path=options['output'],
            source_policy='Production A3 only; PDF text-layer preparation is not ingested as OCR')
        report['colpali_deferred']=options['defer_colpali']
        report['force_reextract']=options['force_reextract']
        report['max_document_pages']=selection_config.get('max_document_pages')
        if options['defer_colpali']:
            report['colpali_deferred_reason']='Remote ngrok endpoint unavailable; separate ColPali indexing required later'
        def save():
            report['updated_at']=timezone.now().isoformat()
            save_state(output,report)
        # A user row lock serializes registration of this batch in the existing tenant.
        for source in sources:
            code=f"EVAL20-{user.company_id}-{source['sha256'][:20]}"
            with transaction.atomic():
                get_user_model().objects.select_for_update().get(pk=user.pk)
                contract=Contract.objects.filter(contract_code=code).first()
                if contract:
                    if contract.company_id!=user.company_id:
                        raise CommandError('Existing document belongs to a different company')
                    version=contract.latest_version
                    stored=version.files.order_by('-uploaded_at').first() if version else None
                    if not stored or file_sha(source['path'])!=source['sha256']:
                        raise CommandError('Source or stored file missing for existing batch document')
                    physical=LayoutExtractionService()._physical_path(stored.file_path)
                    if hashlib.sha256(decrypt_pdf(physical.read_bytes())).hexdigest()!=source['sha256']:
                        raise CommandError('Existing stored PDF hash differs; refusing overwrite')
                else:
                    with source['path'].open('rb') as stream:
                        contract=ContractService().create_and_analyze_contract(code=code,title=source['path'].stem,
                            contract_type='EVALUATION_DOCUMENT',start_date=None,end_date=None,
                            contract_value=None,file_obj=File(stream,name=source['path'].name),company=user.company)
                    version=contract.latest_version
                    version.file_hash=source['sha256']
                    version.save(update_fields=['file_hash'])
            report['documents'].append(dict(alias=source['alias'],source_path=source['source_path'],
                sha256=source['sha256'],contract_id=contract.pk,version_id=version.pk,a3=run_state(version)))
            save()
        if options['register_only']:
            report['status']='REGISTERED_NOT_EXTRACTED'
            save()
            self.stdout.write(str(output))
            return
        report['status']='EXTRACTING'
        save()
        log_path=root/options['log']
        log_path.parent.mkdir(parents=True,exist_ok=True)
        with log_path.open('a',encoding='utf-8',buffering=1) as log:
            handler=logging.StreamHandler(log)
            logging.getLogger().addHandler(handler)
            try:
                with contextlib.redirect_stdout(Tee(log,sys.stdout)),contextlib.redirect_stderr(Tee(log,sys.stderr)):
                    # Short PDFs first; the complete 84-page catalog is retained.
                    import pymupdf
                    page_counts={}
                    for source in sources:
                        with pymupdf.open(source['path']) as pdf:
                            page_counts[source['alias']]=len(pdf)
                    for entry in sorted(report['documents'],key=lambda e:page_counts[e['alias']]):
                        close_old_connections()
                        version=db_read(lambda:ContractVersion.objects.select_related('contract').get(pk=entry['version_id']))
                        state=run_state(version)
                        from ai_extract.models import LayoutExtractionRun
                        if db_read(lambda:LayoutExtractionRun.objects.filter(version=version,status='RUNNING',
                                started_at__gte=timezone.now()-timedelta(hours=24)).exists()):
                            entry['a3']=state
                            entry['batch_error']='A3 run already active; not duplicated'
                            save()
                            continue
                        if state['status']=='COMPLETED' and not options['force_reextract']:
                            entry['a3']=state
                            entry['reused_completed_run']=True
                            save()
                            continue
                        entry['previous_a3']=state
                        report['active_alias']=entry['alias']
                        save()
                        print(json.dumps(dict(event='a3_document_start',alias=entry['alias'],
                            version_id=version.pk,pdf_pages=page_counts[entry['alias']])),flush=True)
                        start=time.perf_counter()
                        try:
                            LayoutExtractionService().extract_version(version)
                        except Exception as error:
                            entry['batch_error']=type(error).__name__+': '+str(error)
                            logging.exception('Production A3 document failed: %s',entry['alias'])
                        entry['elapsed_s']=round(time.perf_counter()-start,3)
                        entry['a3']=run_state(version)
                        save()
                        print(json.dumps(dict(event='a3_document_finish',alias=entry['alias'],
                            status=entry['a3']['status'],elapsed_s=entry['elapsed_s'])),flush=True)
            finally:
                logging.getLogger().removeHandler(handler)
                report.pop('active_alias',None)
                completed=all(e['a3']['status']=='COMPLETED' and not e.get('batch_error') for e in report['documents'])
                report['status']='EXTRACTION_COMPLETED' if completed else 'FINISHED_WITH_FAILURES_OR_PENDING'
                report['graph_or_index_issues']=[e['alias'] for e in report['documents']
                    if e['a3']['status']=='COMPLETED' and (e['a3'].get('graph',{}).get('status')!='ready'
                        or e['a3'].get('dense_status')!='COMPLETED' or e['a3'].get('colpali_status')!='COMPLETED')]
                report['finished_at']=timezone.now().isoformat()
                save()
        self.stdout.write(str(output))

    def check_services(self,require_models=True,require_colpali=True):
        import requests
        from ai_extract.services.contract_pipeline_service import PlanACollectionIndexer
        checks={'a3':settings.RAG_INFERENCE_URL.rstrip('/')+'/rag/health',
                'colpali':PlanACollectionIndexer._colpali_url().rstrip('/')+'/health'}
        if not require_colpali:
            checks.pop('colpali',None)
        if not require_models:
            checks={}
        for name,url in checks.items():
            try:
                response=requests.get(url,headers={'ngrok-skip-browser-warning':'true'},timeout=12)
                self.stdout.write(f'{name}: HTTP {response.status_code}')
                if not response.ok:
                    raise CommandError(f'{name} is unavailable; no PDFs imported')
            except requests.RequestException as error:
                raise CommandError(f'{name}: {type(error).__name__}; no PDFs imported') from error
