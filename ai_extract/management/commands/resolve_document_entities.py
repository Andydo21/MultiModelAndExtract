"""Durable identity work within each document, outside the A2 request path."""
import json
import logging
import time
from django.core.management import call_command
from django.core.management.base import BaseCommand, CommandError
from ai_extract.models import LayoutExtractionRun
from ai_extract.repositories import document_graph_repository as repository, layout_repository
from ai_extract.services.layout_service import enrich_entity_graph
from ai_extract.services.document_graph_service import validate_prepared


class Command(BaseCommand):
    help = 'Resolve repeated entity mentions within each document snapshot; retry without redoing OCR.'

    def add_arguments(self, parser):
        parser.add_argument('--watch',action='store_true',help='Keep processing newly completed snapshots.')
        parser.add_argument('--run-id',type=int,help='Queue a completed run through the normal graph synchronization flow.')

    def handle(self,*args,**options):
        repository.entity_schema()
        if options['run_id']:
            call_command('sync_document_graph',run_id=options['run_id'],stdout=self.stdout)
        try:
            while True:
                # OCR/NER and identity share one small GPU. Let extraction finish
                # before the background identity worker starts swapping models.
                if options['watch'] and layout_repository.extraction_in_progress():
                    time.sleep(5)
                    continue
                result=self.process_next()
                if result is None:
                    if not options['watch']:return
                    time.sleep(5);continue
                if result['status']!='complete':
                    if not options['watch']:raise CommandError('Identity job remains pending; no identity was inferred from similarity alone.')
                    time.sleep(30)
        except KeyboardInterrupt:
            return

    def process_next(self):
        with layout_repository.workflow_slot() as available:
            if not available:return None
            scope=repository.claim_entity_job()
            if scope is None:return None
            run=LayoutExtractionRun.objects.select_related('version__contract').filter(
                pk=scope['run_id'],status='COMPLETED',version_id=scope['version_id'],
                version__contract__company_id=scope['company_id']).first()
            try:
                latest=layout_repository.latest_run(run.version,completed=True) if run else None
                if (not run or not latest or latest.pk!=run.pk
                        or run.structure.get('source_fingerprint')!=scope['fingerprint']
                        or run.structure.get('graph_index',{}).get('status')!='ready'):
                    result=dict(status='complete',skipped='retired_or_unavailable_source')
                else:
                    # Legacy queued jobs must use the same pre-publication
                    # models as HTTP extraction; never analyze saved graph nodes.
                    enrich_entity_graph(run)
                    validate_prepared(run.structure)
                    identity=run.structure['identity_resolution']
                    result={k:identity[k] for k in ('status','model','revision','mentions',
                        'linked','unresolved','canonical_entities')}
            except Exception as error:
                logging.getLogger(__name__).exception('Entity resolution failed for run %s',scope['run_id'])
                result=dict(status='unavailable',error_type=type(error).__name__)
            repository.finish_entity_job(scope,result)
            self.stdout.write(json.dumps(dict(event='entity_resolution',run=scope['run_id'],**result),ensure_ascii=False))
            self.stdout.flush()
            return result
