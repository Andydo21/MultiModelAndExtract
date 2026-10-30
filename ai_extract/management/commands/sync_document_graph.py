"""Explicitly backfill a completed A3 run without changing contract versions."""
from django.core.management.base import BaseCommand, CommandError
from ai_extract.models import LayoutExtractionRun
from ai_extract.repositories import layout_repository
from ai_extract.services.document_graph_service import sync_graph


class Command(BaseCommand):
    help = 'Sync a completed A3 run into Neo4j; optionally extract source-backed knowledge.'

    def add_arguments(self, parser):
        parser.add_argument('--run-id', type=int, required=True)
        parser.add_argument('--analyze-relations', action='store_true',
                            help='Extract entities, local relations and whole-document GLiNER reference proposals.')

    def handle(self, *args, **options):
        run = LayoutExtractionRun.objects.select_related('version__contract').filter(
            pk=options['run_id'], status='COMPLETED').first()
        if not run or not run.structure or not run.structure.get('source_fingerprint'):
            raise CommandError('A completed model structure with a source fingerprint is required.')
        from ai_extract.services.document_structure_service import source_fingerprint
        if source_fingerprint(layout_repository.structure_records(run)) != run.structure['source_fingerprint']:
            raise CommandError('Sources changed; rebuild the structure before graph synchronization.')
        from ai_extract.services.document_structure_service import attach_reading_streams
        run.structure=attach_reading_streams(run.structure)
        if options['analyze_relations']:
            from ai_extract.services.layout_service import enrich_entity_graph
            with layout_repository.workflow_slot() as available:
                if not available:raise CommandError('Another document workflow is active.')
                enrich_entity_graph(run)
        else:
            run.structure['graph_index'] = sync_graph(run)
            layout_repository.save_run(run, ['structure'])
        result = run.structure['graph_index']
        self.stdout.write(str(dict(graph=result)))
        review=run.structure.get('structure_review',{})
        self.stdout.write(str(dict(structure_review={k:review[k] for k in ('status','counts','relations_total','errors') if k in review})))
        if result['status'] != 'ready':
            raise CommandError('Graph sync did not complete; inspect configuration and service availability.')
        from ai_extract.services.document_graph_service import cleanup_old_run_graphs
        cleanup_old_run_graphs(run)
