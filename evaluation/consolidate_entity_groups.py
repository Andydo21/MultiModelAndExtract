"""Prepare/apply source-anchored merges for the authorized 20-document corpus.

Execute through manage.py shell. GLiNER2 must approve each candidate group.
Only RESOLVES_TO and orphan identity registry nodes are changed.
"""
import hashlib
import json
from pathlib import Path
from uuid import uuid5,NAMESPACE_URL
from django.utils import timezone
from ai_extract.repositories.document_graph_repository import connection
from ai_extract.services.identity_source_anchors import source_identity_anchor,normalized,source_identity_key
from evaluation.batch_io import save_state

directory=Path('/app/evaluation/benchmarks/corpus20')


def prepare():
    manifest=json.loads((directory/'a3_ingestion.json').read_text(encoding='utf-8-sig'))
    assert manifest['company_id']==5
    runs=[e['a3']['run_id'] for e in manifest['documents']]
    with connection() as s:
        rows=s.run('''MATCH (s:A3Snapshot {company_id:5,completed:true}) WHERE s.run_id IN $runs
          MATCH (m:A3Node {snapshot_key:s.key,category:'entity'})<-[:MENTIONS]-(p:A3Node)
          WHERE m.entity_type IN ['PRODUCT','ORGANIZATION'] AND p.category='evidence'
          WITH m.entity_type AS kind,m.name AS name,p.text AS context,
            collect(DISTINCT {key:m.key,snapshot_key:s.key,run_id:s.run_id,version_id:s.version_id,evidence_id:p.leaf_id}) AS members
          WHERE size(members)>1 RETURN kind,name,context,members ORDER BY name,context''',runs=runs).data()
    groups=[]
    for row in rows:
        if len({r['key'] for r in row['members']})<2:continue
        anchor=source_identity_anchor(row['name'],row['kind'],row['context'])
        if not anchor:continue
        identity=hashlib.sha256(json.dumps([row['kind'],row['name'],row['context']],ensure_ascii=False).encode()).hexdigest()
        row.update(id=identity,anchor=anchor,entity_type=row['kind'],
            left=dict(name=row['name'],context=row['context']),right=dict(name=row['name'],context=row['context']))
        groups.append(row)
    save_state(directory/'entity_merge_groups.json',dict(company_id=5,prepared_at=timezone.now().isoformat(),groups=groups))
    print(json.dumps(dict(groups=len(groups),mentions=sum(len({m['key'] for m in g['members']}) for g in groups),names=[g['name'] for g in groups])),flush=True)


def apply():
    audit=directory/'entity_merge_audit.json'
    if audit.exists() and json.loads(audit.read_text()).get('status')=='APPLIED':
        raise RuntimeError('Merge already applied; retain the original before-image audit')
    source=directory/'entity_merge_groups.json'
    prepared=json.loads(source.read_text())
    decisions=json.loads((directory/'entity_merge_group_decisions.json').read_text())
    assert decisions['source_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
    judgments={r['id']:r for r in decisions['results']}
    report=dict(started_at=timezone.now().isoformat(),model=decisions['model'],revision=decisions['revision'],groups=[],graph_source_nodes_preserved=True)
    with connection() as session:
        def write(tx):
            report['groups']=[]  # execute_write may retry the transaction.
            tx.run('MERGE (s:A3EntityScope {company_id:5}) SET s.updated_at=timestamp()').consume()
            baseline=tx.run('MATCH (n:A3Node) RETURN count(n) AS nodes').single()['nodes']
            mentions_before=tx.run('MATCH (:A3Node)-[r:MENTIONS]->(:A3Node) RETURN count(r) AS count').single()['count']
            for group in prepared['groups']:
                decision=judgments[group['id']]
                if decision['status']!='same':continue
                assert source_identity_anchor(group['name'],group['kind'],group['context'])==group['anchor']
                keys=sorted({m['key'] for m in group['members']})
                snapshots=sorted({m['snapshot_key'] for m in group['members']})
                rows=tx.run('''MATCH (s:A3Snapshot {company_id:5,completed:true}) WHERE s.key IN $snapshots
                  MATCH (m:A3Node {snapshot_key:s.key,category:'entity'})<-[:MENTIONS]-(p:A3Node)
                  WHERE m.key IN $keys AND m.name=$name AND m.entity_type=$kind AND p.text=$context
                  RETURN DISTINCT m.key AS key''',snapshots=snapshots,keys=keys,name=group['name'],kind=group['kind'],context=group['context']).data()
                assert sorted(r['key'] for r in rows)==keys,'Source group changed; merge aborted'
                old=tx.run('''MATCH (m:A3Node) WHERE m.key IN $keys
                  OPTIONAL MATCH (m)-[r:RESOLVES_TO]->(e:A3Entity {company_id:5})
                  RETURN m.key AS mention_key,properties(r) AS relationship,properties(e) AS identity''',keys=keys).data()
                target=source_identity_key(5,group['name'],group['kind'],group['anchor'])
                embedding=next((r['identity'].get('embedding') for r in old if r['identity'] and r['identity'].get('embedding')),None)
                lookup=hashlib.sha256(f"5:{group['kind']}".encode()).hexdigest()
                tx.run('''MERGE (e:A3Entity {key:$key}) ON CREATE SET e.company_id=5,e.entity_type=$kind,e.name=$name,e.aliases=[$name]
                  SET e.identity_model=$model,e.identity_revision=$revision,e.identity_review_status='unreviewed',
                    e.lookup_scope=$lookup,e.embedding=coalesce(e.embedding,$embedding),e.identity_level=$level,
                    e.identity_anchors=CASE WHEN $anchor IN coalesce(e.identity_anchors,[]) THEN e.identity_anchors ELSE coalesce(e.identity_anchors,[])+$anchor END,
                    e.aliases=CASE WHEN $name IN e.aliases THEN e.aliases ELSE e.aliases+$name END ''',
                  key=target,kind=group['kind'],name=group['name'],anchor=json.dumps(group['anchor']),model=decisions['model'],revision=decisions['revision'],lookup=lookup,embedding=embedding,
                  level='registered_brand' if group['kind']=='PRODUCT' else 'named_organization').consume()
                tx.run('''MATCH (m:A3Node) WHERE m.key IN $keys
                  OPTIONAL MATCH (m)-[old:RESOLVES_TO]->(:A3Entity {company_id:5}) DELETE old
                  WITH DISTINCT m MATCH (e:A3Entity {key:$target,company_id:5})
                  MERGE (m)-[r:RESOLVES_TO]->(e)
                  SET r.status='model_linked',r.origin='gliner2_source_anchored_identity',r.review_status='unreviewed',
                      r.model=$model,r.revision=$revision,r.decision_id=$decision,
                      m.identity_resolution_status='model_linked' ''',keys=keys,target=target,model=decisions['model'],revision=decisions['revision'],decision=group['id']).consume()
                old_keys=list({r['identity']['key'] for r in old if r['identity'] and r['identity']['key']!=target})
                tx.run('''MATCH (e:A3Entity {company_id:5}) WHERE e.key IN $keys
                  AND NOT EXISTS {MATCH (:A3Node)-[:RESOLVES_TO]->(e)} DETACH DELETE e''',keys=old_keys).consume()
                count=tx.run('MATCH (m:A3Node)-[:RESOLVES_TO]->(e:A3Entity {key:$target}) WHERE m.key IN $keys RETURN count(DISTINCT m) AS mentions,count(DISTINCT e) AS identities',target=target,keys=keys).single()
                assert count['mentions']==len(keys) and count['identities']==1
                report['groups'].append(dict(id=group['id'],name=group['name'],kind=group['kind'],anchor=group['anchor'],target=target,mentions=len(keys),before=old))
            assert tx.run('MATCH (n:A3Node) RETURN count(n) AS nodes').single()['nodes']==baseline
            assert tx.run('MATCH (:A3Node)-[r:MENTIONS]->(:A3Node) RETURN count(r) AS count').single()['count']==mentions_before
            report['source_nodes_before']=report['source_nodes_after']=baseline
            report['source_mentions_before']=report['source_mentions_after']=mentions_before
            # Durable before-image on the host must exist before graph commit.
            save_state(directory/'entity_merge_audit.json',dict(report,status='PREPARED_FOR_COMMIT'))
        session.execute_write(write)
    report.update(status='APPLIED',finished_at=timezone.now().isoformat())
    save_state(directory/'entity_merge_audit.json',report)
    print(json.dumps(dict(status='APPLIED',groups=len(report['groups']),mentions=sum(g['mentions'] for g in report['groups']),names=[g['name'] for g in report['groups']])),flush=True)
