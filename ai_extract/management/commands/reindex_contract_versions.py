from django.core.management.base import BaseCommand,CommandError
from ai_extract.services import layout_service


class Command(BaseCommand):
    help='Reindex the latest completed extraction per version; preserve source files.'

    def add_arguments(self,parser):
        scope=parser.add_mutually_exclusive_group()
        scope.add_argument('--version-id',type=int)
        scope.add_argument('--run-id',type=int,help='Use a specific completed run with --colpali-only.')
        mode=parser.add_mutually_exclusive_group()
        mode.add_argument('--dense-only',action='store_true',help='Retry only text vectors; preserve OCR, graph and visual index.')
        mode.add_argument('--colpali-only',action='store_true',help='Backfill stored figure images; preserve OCR, entities and dense vectors.')

    def handle(self,**options):
        if any(options[key] is not None and options[key]<=0 for key in ('run_id','version_id')):
            raise CommandError('Run/version ID must be positive')
        if options['run_id'] is not None:
            if not options['colpali_only']:raise CommandError('--run-id requires --colpali-only')
            from ai_extract.models import LayoutExtractionRun
            try:
                run=LayoutExtractionRun.objects.get(pk=options['run_id'])
                result=layout_service.backfill_colpali(run)
            except (LayoutExtractionRun.DoesNotExist,ValueError,RuntimeError) as error:
                raise CommandError(str(error)) from error
            self.stdout.write(f'Run {run.pk}: ColPali={result["status"]} ({result["points"]}), graph={result["graph"]["status"]}')
            if result['status'] in {'FAILED','PARTIAL'} or result['points'] and result['graph']['status']!='ready':
                raise CommandError(result['error'] or 'ColPali vectors or graph links are incomplete')
            return
        failed=[]
        try:
            for run in layout_service.reindex_versions(options['version_id'],dense_only=options['dense_only'],colpali_only=options['colpali_only']):
                self.stdout.write(f'Version {run.version_id}: dense={run.dense_index_status} ({run.dense_points_count}), visual={run.colpali_index_status} ({run.colpali_points_count})')
                if options['colpali_only'] and (run.colpali_index_status in {'FAILED','PARTIAL'}
                        or run.colpali_points_count and run.structure.get('colpali_backfill',{}).get('graph',{}).get('status')!='ready'):
                    failed.append(str(run.pk))
        except (ValueError,RuntimeError) as error:raise CommandError(str(error)) from error
        if failed:raise CommandError('ColPali backfill incomplete for runs: '+', '.join(failed))
