import json
import shutil
from pathlib import Path
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from ai_extract.models import LayoutExtractionRun


def validate_page(page, require_labels=False):
    blocks = {b['id']: b for b in page['blocks']}
    regions = {r['id']: r for r in page['regions']}
    if len(blocks) != len(page['blocks']) or len(regions) != len(page['regions']):
        raise ValueError('Duplicate source IDs')
    for item in list(blocks.values()) + list(regions.values()):
        box = item['bbox']
        if len(box) != 4 or any(not isinstance(v, (int, float)) or not 0 <= v <= 1000 for v in box) or not box[0] < box[2] or not box[1] < box[3]:
            raise ValueError('Invalid normalized source bbox')
    for pair in page['pairs']:
        if pair['text_id'] not in blocks or pair['region_id'] not in regions:
            raise ValueError('Pair refers to missing source')
        if pair.get('label') not in {None, 0, 1} or require_labels and pair.get('label') is None:
            raise ValueError('Manual positive/negative labels are required')
        if pair.get('label') is not None and not pair.get('reviewer', '').strip():
            raise ValueError('Labeled pairs require reviewer provenance')
    return page

def build_page(run, page, image):
    blocks = [dict(id=f'block:{b.pk}', text=b.text, bbox=b.bbox, layout_type=b.layout_type)
              for b in page.blocks.all() if b.text.strip() and len(b.bbox) == 4]
    regions = [dict(id=f'figure:{b.pk}', bbox=b.bbox, origin='observed_layout')
               for b in page.blocks.all() if b.layout_type in {'figure', 'picture', 'image', 'diagram', 'chart'} and len(b.bbox) == 4]
    pairs = [dict(text_id=b['id'], region_id=r['id'], label=None, reviewer='')
             for b in blocks for r in regions]
    return validate_page(dict(schema='layout-caption-pairs-v1', document_id=f'version:{run.version_id}',
        run_id=run.pk, page_number=page.page_number, image=image, blocks=blocks, regions=regions, pairs=pairs))


class Command(BaseCommand):
    help = 'Export source pages and unlabeled text/figure pairs for human review.'
    def add_arguments(self, parser):
        parser.add_argument('--version-id', type=int, required=True)
        parser.add_argument('--output', required=True)
    def handle(self, **options):
        run = LayoutExtractionRun.objects.filter(version_id=options['version_id'], status='COMPLETED').first()
        if not run:
            raise CommandError('No completed source extraction')
        folder = Path(options['output']).resolve()
        folder.mkdir(parents=True, exist_ok=True)
        (folder/'images').mkdir(exist_ok=True)
        root = Path(settings.MEDIA_ROOT).resolve()
        rows = []
        for page in run.pages.prefetch_related('blocks').order_by('page_number'):
            source = (root/page.image_path).resolve()
            if not page.image_path or not source.is_relative_to(root) or not source.is_file():
                continue
            name = f'run{run.pk}-p{page.page_number}.png'
            shutil.copyfile(source, folder/'images'/name)
            rows.append(build_page(run, page, 'images/'+name))
        if not rows:
            raise CommandError('No rendered source pages')
        (folder/'pages.jsonl').write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows)+'\n', encoding='utf-8')
        self.stdout.write(f'Exported {len(rows)} pages; {sum(len(r["pairs"]) for r in rows)} UNLABELED pairs. Missing figure regions must be annotated manually.')
