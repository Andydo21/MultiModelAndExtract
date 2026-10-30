"""Isolated E1 representation and E2 expansion adapters, including production bridge."""
import json
import re
from collections import deque

from .io import canonical

E1_FORMATS = ('A','B','C','D','E')
E2_MODES = {'A':'none', 'B':'previous_next', 'C':'structural', 'D':'scope_aware'}


def table_text(item, markdown=True):
    table = item.get('table')
    if not table:
        return ''
    rows = [table['columns'], *table['rows']]
    if not markdown:
        return '\n'.join(' ; '.join(map(str,row)) for row in rows)
    def cell(value):
        return str(value).replace('|','\\|').replace('\n','<br>')
    return '\n'.join('| '+' | '.join(map(cell,row))+' |'
                     for row in [rows[0], ['---']*len(rows[0]), *rows[1:]])


def table_replaces_text(item):
    """Production pods show cells instead of OCR text only when cells reproduce that text."""
    table = item.get('table')
    if not table:
        return False
    flat = ''.join(str(c) for row in [table['columns'], *table['rows']] for c in row)
    return re.sub(r'\s+','',flat) == re.sub(r'\s+','',item.get('text',''))


def production_pods(sources, graph):
    """Call real NewIdea rendering lazily; no renderer impersonation or DB writes.

    Source order is a controlled variable: refuse grouping-induced reorder.
    Requires the actual production source envelopes, not invented fields.
    """
    from NewIdea.services.knowledge_answer_service import answer_source_pods, render_source_pods
    if not all(item.get('production_source') for item in sources):
        raise ValueError('Production renderer requires actual production_source envelopes')
    originals = [item['production_source'] for item in sources]
    if any(item['evidence_id'] != src.get('evidence_id') or item['text'] != src.get('text')
           for item, src in zip(sources, originals)):
        raise ValueError('Production source identity/text mismatch')
    pods = answer_source_pods(originals, graph or {})
    actual = [source['id'] for pod in pods['pods'] for source in pod['sources']]
    if actual != [item['evidence_id'] for item in sources]:
        raise ValueError('Production grouping changes source order; freeze compatible input ordering first')
    return render_source_pods(pods)


def render(evidence, variant, *, renderer='canonical', graph=None):
    if variant not in E1_FORMATS:
        raise ValueError('Unknown E1 format')
    if renderer not in ('canonical','production'):
        raise ValueError('Unknown pod renderer')
    if variant == 'E' and renderer == 'production':
        return production_pods(evidence, graph)
    parts = []
    for item in evidence:
        if variant == 'B':
            # Deliberately serialize the SAME content fields as A/C; full metadata
            # here would confound JSON syntax with additional evidence.
            fields = {k:item.get(k) for k in ('evidence_id','text','table')}
            if table_replaces_text(item):
                fields['text'] = None
            parts.append(canonical(fields))
            continue
        prefix = '['+item['evidence_id']+']'
        if variant in ('D','E'):
            prefix += '\n'+' > '.join(item.get('breadcrumb',[]))
        if variant == 'E':
            prefix += '\n'+canonical({k:item.get(k) for k in
                                      ('document_id','version','page','region','scope','links')})
        body = item['text']
        if table_replaces_text(item):
            body = table_text(item, markdown=variant!='A')
        elif item.get('table'):
            body += '\n'+table_text(item, markdown=variant!='A')
        parts.append(prefix+'\n'+body)
    return '\n\n'.join(parts)


def expand(seeds, nodes, edges, mode, token_count, budget, hops=1):
    mode = E2_MODES.get(mode, mode)
    if mode not in E2_MODES.values() or hops < 0:
        raise ValueError('Unknown expansion mode or invalid hop limit')
    by_id = {n['evidence_id']:n for n in nodes}
    if len(by_id)!=len(nodes) or len(set(seeds))!=len(seeds):
        raise ValueError('Duplicate evidence ID')
    selected = [by_id[eid] for eid in seeds]
    if token_count(selected)>budget:
        raise ValueError('Seed exceeds budget; no source truncated')
    seen, queue, trace = set(seeds), deque((eid,0) for eid in seeds), []
    while queue and mode!='none':
        source, depth = queue.popleft()
        if depth >= hops:
            continue
        for index, edge in enumerate(edges):
            if edge['source']!=source:
                continue
            left, right = by_id[source], by_id.get(edge['target'])
            reason = None
            if right is None:
                reason = 'missing_target'
            elif edge['target'] in seen:
                reason = 'already_selected'
            elif any(left.get(k) is None or left[k]!=right.get(k) for k in ('document_id','version')):
                reason = 'document_version_boundary'
            elif mode=='previous_next' and edge['type'] not in ('PREVIOUS','NEXT'):
                reason = 'edge_type'
            elif mode in ('structural','scope_aware') and edge['type'] not in (
                    'PARENT','CHILD','TABLE_HEADER','FOOTNOTE','CONDITION','REFERENCE'):
                reason = 'edge_type'
            elif mode=='scope_aware' and (not left.get('scope') or left['scope']!=right.get('scope')):
                reason = 'scope_boundary'
            elif token_count(selected+[right])>budget:
                reason = 'token_budget'
            trace.append(dict(edge_index=index, edge=edge, accepted=reason is None,
                              reason=reason or 'admitted', depth=depth+1))
            if reason is None:
                selected.append(right)
                seen.add(edge['target'])
                queue.append((edge['target'],depth+1))
    return selected, trace


SYSTEM = ('Answer every supported part of the question directly and in Vietnamese, using only the supplied evidence. '
    'Give a concise, complete answer; if a requested detail is unavailable, state that limit. '
    'Keep subject, attribute, value and condition together. '
    'Cite evidence IDs. State when evidence is insufficient. Return JSON with answer and claims; '
    'each claim has subject, attribute, value, condition and evidence_ids.')


def messages(question, evidence, variant, **options):
    return [dict(role='system',content=SYSTEM),dict(role='user',content=question+'\n\nEVIDENCE\n'+render(evidence,variant,**options))]
