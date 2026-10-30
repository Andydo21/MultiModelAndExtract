"""Read-only SQL/Neo4j verification against the saved source-anchored merge audit.

Run with manage.py shell; writes a new timestamped verification artifact only.
"""
import json
from pathlib import Path
from django.utils import timezone
from ai_extract.models import LayoutExtractionRun
from ai_extract.repositories.document_graph_repository import connection
from evaluation.batch_io import save_state


def check():
    directory = Path('/app/evaluation/benchmarks/corpus20')
    audit = json.loads((directory / 'entity_merge_audit.json').read_text())
    manifest = json.loads((directory / 'a3_ingestion.json').read_text())
    versions = [d['version_id'] for d in manifest['documents']]
    result = dict(checked_at=timezone.now().isoformat(),
                  historical_merge_status=audit.get('status'), groups=[], documents=[])
    for version in versions:
        run = LayoutExtractionRun.objects.filter(version_id=version).order_by('-pk').first()
        result['documents'].append(dict(version_id=version, run_id=run.pk,
            status=run.status, pages=run.total_pages, dense=run.dense_index_status,
            colpali=run.colpali_index_status, graph=run.structure.get('graph_index', {})))
    with connection() as session:
        for group in audit['groups']:
            row = session.run('''MATCH (m:A3Node)-[:RESOLVES_TO]->(e:A3Entity {key:$target})
                WHERE m.key IN $keys
                RETURN count(DISTINCT m) AS mentions,count(DISTINCT e) AS identities''',
                target=group['target'], keys=[r['mention_key'] for r in group['before']]).single()
            result['groups'].append(dict(name=group['name'], target=group['target'],
                expected=group['mentions'], **dict(row)))
        result['current_shared_identities'] = session.run('''
            MATCH (s:A3Snapshot {company_id:5,completed:true}) WHERE s.version_id IN $versions
            MATCH (m:A3Node {snapshot_key:s.key,category:'entity'})-[:RESOLVES_TO]->(e:A3Entity)
            WITH e, count(DISTINCT m) AS mentions,collect(DISTINCT s.version_id) AS versions
            WHERE mentions>1 RETURN e.key AS key,e.name AS name,mentions,versions,
                e.identity_model AS model ORDER BY mentions DESC''', versions=versions).data()
    result['historical_links_intact'] = all(g['mentions'] == g['expected'] and
        g['identities'] == 1 for g in result['groups'])
    path = directory / ('entity_verification_' + timezone.now().strftime('%Y%m%dT%H%M%SZ') + '.json')
    save_state(path, result)
    print(json.dumps(dict(path=str(path), documents=len(versions),
        historical_links_intact=result['historical_links_intact'],
        groups=result['groups'], shared_identities=result['current_shared_identities'])))


check()
