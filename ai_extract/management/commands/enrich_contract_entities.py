from django.core.management.base import BaseCommand,CommandError
from ai_extract.models import LayoutExtractionRun
from ai_extract.services.layout_service import enrich_entity_graph


class Command(BaseCommand):
    help='Run GLiNER and save entity mentions to the existing Neo4j source graph; preserve OCR and vectors.'

    def add_arguments(self,parser):
        parser.add_argument('--run-id',type=int,required=True)

    def handle(self,**options):
        try:
            run=LayoutExtractionRun.objects.select_related('version__contract').get(pk=options['run_id'])
            enrich_entity_graph(run)
        except (LayoutExtractionRun.DoesNotExist,ValueError,RuntimeError) as error:
            raise CommandError(str(error)) from error
        self.stdout.write(self.style.SUCCESS(f'Run {run.id}: GLiNER={run.structure["knowledge_graph"]["status"]}, '
                                            f'entities={len(run.structure["knowledge_graph"]["entities"])}, Neo4j=ready'))
