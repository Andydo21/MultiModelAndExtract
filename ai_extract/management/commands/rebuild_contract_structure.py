from django.core.management.base import BaseCommand
from ai_extract.services import layout_service


class Command(BaseCommand):
    help = 'Analyze stored source blocks and images with configured structure provider; then reindex for consistent retrieval.'

    def add_arguments(self, parser):
        parser.add_argument('--version-id', type=int, help='Limit analysis to one contract version.')

    def handle(self, **options):
        count = layout_service.rebuild_structures(options['version_id'])
        self.stdout.write(f'Rebuilt {count} extracted contract structures.')
