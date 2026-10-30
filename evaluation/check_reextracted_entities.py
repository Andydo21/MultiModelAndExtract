"""Read-only entity audit for the latest completed re-extractions."""
import json
from pathlib import Path
from django.utils import timezone
from ai_extract.models import LayoutExtractionRun
from ai_extract.repositories.document_graph_repository import connection
from evaluation.batch_io import save_state


def check():
    rows = []
    runs = LayoutExtractionRun.objects.filter(pk__gte=76, status='COMPLETED').order_by('pk')
    with connection() as session:
        for run in runs:
            snapshots = session.run('''MATCH (s:A3Snapshot {run_id:$run,completed:true})
                RETURN s.key AS key,s.entity_resolution_state AS state,
                  s.entity_resolution_attempts AS attempts,s.entity_resolution_result AS result''',
                run=run.pk).data()
            for snapshot in snapshots:
                groups = session.run('''MATCH (m:A3Node {snapshot_key:$key,category:'entity'})
                    OPTIONAL MATCH (m)-[r:RESOLVES_TO]->(e:A3Entity)
                    WITH m.name AS name,m.entity_type AS kind,count(DISTINCT m) AS mentions,
                      count(DISTINCT e) AS identities,collect(DISTINCT e.key) AS keys,
                      collect(DISTINCT r.origin) AS origins
                    RETURN name,kind,mentions,identities,keys,origins
                    ORDER BY mentions DESC,name''', key=snapshot['key']).data()
                rows.append(dict(run_id=run.pk, version_id=run.version_id,
                    snapshot=snapshot, groups=groups))
    report = dict(checked_at=timezone.now().isoformat(), runs=rows)
    path = Path('/app/evaluation/benchmarks/corpus20') / (
        'reextracted_entities_' + timezone.now().strftime('%Y%m%dT%H%M%SZ') + '.json')
    save_state(path, report)
    print(json.dumps(dict(path=str(path),runs=[dict(run_id=r['run_id'],version_id=r['version_id'],
        snapshot=r['snapshot'],repeated_named_entities=[g for g in r['groups']
            if g['kind'] in {'PRODUCT','ORGANIZATION'} and g['mentions']>1]) for r in rows]),ensure_ascii=False))


check()
