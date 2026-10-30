"""Index existing saved expert knowledge; never rerun Qwen or document OCR."""
from django.core.management.base import BaseCommand, CommandError
from django.contrib.auth import get_user_model
from NewIdea.services.search_evidence_service import visible_text_records
from ai_extract.services.contract_pipeline_service import PlanACollectionIndexer


class Command(BaseCommand):
    help = 'Backfill expert Q/A, semantic units and approved cases into the A3 BGE collection.'

    def add_arguments(self, parser):
        parser.add_argument('--username', required=True, help='Use this account\'s authorized corpus.')
        parser.add_argument('--dry-run', action='store_true')

    def handle(self, *args, **options):
        user = get_user_model().objects.filter(username=options['username'], is_active=True).first()
        if not user:
            raise CommandError('Active account not found')
        records = visible_text_records(user)
        self.stdout.write(f'Authorized text records: {len(records)}')
        if options['dry_run']:
            return
        indexer = PlanACollectionIndexer()
        for offset in range(0, len(records), 16):
            result = indexer.index_text_records(records[offset:offset+16])
            if result.status != 'COMPLETED':
                raise CommandError(f'Knowledge indexing failed: {result.error}')
            self.stdout.write(f'Indexed {min(offset+16, len(records))}/{len(records)}')
