"""Deterministic lexical retrieval and evidence checks; no model loading."""
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter


def tokens(text):
    text = unicodedata.normalize('NFC', str(text)).casefold()
    return re.findall(r'[^\W_]+(?:[-./][^\W_]+)*', text, re.UNICODE)


def identifiers(text):
    return {t for t in tokens(text) if any(c.isdigit() for c in t)
            and any(c.isalpha() for c in t)}


def query_policy(query):
    words = set(tokens(query))
    # Equal baseline until a held-out contract benchmark supports tuning.
    # Intent remains descriptive; it does not silently change weights.
    weights = {'dense': 1.0, 'keyword': 1.0, 'colpali': 1.0, 'graph': 1.0}
    intent = 'semantic'
    if words & {'hình', 'vẽ', 'diagram', 'drawing', 'connector', 'sơ', 'ảnh'}:
        intent = 'visual'
    if identifiers(query):
        intent = 'exact_identifier'
    return {'intent': intent, 'weights': weights, 'version': 'equal-baseline-v2'}


def fusion_rank(stream_name, record, ordinal):
    """ColPali ranks pages; selected chunks inherit their page's rank."""
    page_rank = record.get('colpali_page_rank')
    if stream_name == 'colpali' and type(page_rank) is int and page_rank > 0:
        return page_rank
    return ordinal


def lexical_search(query, records, limit=30):
    """BM25 with token boundaries, IDF and length normalization over all chunks."""
    records = [r for r in records if not r.get('retrieval_excluded')]
    terms = set(tokens(query))
    if not terms or not records:
        return []
    counts = [Counter(tokens(r.get('text', '') + ' ' + r.get('original_name', '')))
              for r in records]
    lengths = [sum(c.values()) for c in counts]
    avg_length = sum(lengths) / len(lengths) or 1
    frequencies = Counter(t for c in counts for t in terms if t in c)
    ranked = []
    for record, counts_i, length in zip(records, counts, lengths):
        score = 0.0
        for term in terms:
            tf = counts_i[term]
            if tf:
                idf = math.log(1 + (len(records) - frequencies[term] + .5)
                               / (frequencies[term] + .5))
                score += idf * tf * 2.5 / (tf + 1.5 * (.25 + .75 * length / avg_length))
        if score > 0:
            ranked.append(dict(record, lexical_score=round(score, 6), score=score))
    return sorted(ranked, key=lambda r: r['lexical_score'], reverse=True)[:limit]


def evidence_identity(item):
    doc = item.get('document_id') or item.get('original_name') or 'unknown'
    page = item.get('page_number', 1)
    chunk = item.get('chunk_id')
    if chunk is None:
        chunk = hashlib.sha256(str(item.get('text', '')).encode()).hexdigest()[:16]
    return f'{doc}:{page}:{chunk}'


def gate_evidence(query, citations):
    """Heuristic policy signals, explicitly not a probability of correctness."""
    useful = [c for c in citations if str(c.get('text') or '').strip()
              and (c.get('layout_type') != 'neo4j_graph_node' or c.get('document_id'))]
    required = identifiers(query)
    present = set().union(*(set(tokens(c.get('text', ''))) for c in useful)) if useful else set()
    missing = sorted(required - present)
    scores = [float(c.get('rerank_score') or 0) for c in useful]
    accepted = bool(useful) and not missing and max(scores, default=0) >= 35
    return {'accepted': accepted, 'reason': 'supported_candidates' if accepted else
            ('no_text_evidence' if not useful else 'missing_identifier' if missing else 'weak_ranking'),
            'missing_identifiers': missing, 'evidence_count': len(useful),
            'score_kind': 'heuristic_ranking_signal', 'policy_version': 'hackathon-v1'}


def check_citation_ids(answer, citations):
    valid = {c['evidence_id'] for c in citations}
    used = set(re.findall(r'\[(E\d+)\]', answer))
    return {'valid': bool(used) and used <= valid,
            'unknown_ids': sorted(used - valid), 'used_ids': sorted(used),
            'scope': 'identifier_validation_only'}


def source_excerpt(citations):
    lines = ['Chưa thể xác minh câu trả lời tổng hợp. Các đoạn nguồn để đối chiếu:']
    for c in citations[:4]:
        location = c.get('source_location') or f"trang {c.get('page_number', 1)}"
        lines.append(f"[{c['evidence_id']}] {c.get('original_name', 'Tài liệu')} — "
                     f"{location}\n{c.get('text', '')[:1200]}")
    return '\n\n'.join(lines)


def evidence_window(query, text, max_chars=2000):
    """A bounded, verbatim source window containing the best exact query term."""
    if len(text) <= max_chars:
        return text
    terms = sorted(set(tokens(query)), key=lambda t: (t in identifiers(query), len(t)), reverse=True)
    hit = None
    for term in terms:
        hit = re.search(r'(?<!\w)' + re.escape(term) + r'(?!\w)', text, re.IGNORECASE)
        if hit:
            break
    start = max(0, hit.start() - max_chars // 3) if hit else 0
    start = min(start, max(0, len(text) - max_chars))
    return text[start:start + max_chars]


class LexicalOnlyReranker:
    """Explicit degraded mode for demos without neural runtime dependencies."""
    @staticmethod
    def reciprocal_rank_fusion(ranked_lists, k=60, weights=None):
        weights = weights or {}
        fused = {}
        for name, stream in ranked_lists.items():
            seen = set()
            for rank, record in enumerate(stream, 1):
                key = evidence_identity(record)
                if key in seen:
                    continue
                seen.add(key)
                item = fused.setdefault(key, dict(record, rrf_score=0, retrieval_streams=[]))
                item['rrf_score'] += weights.get(name, 1) / (k + fusion_rank(name,record,rank))
                if name == 'colpali':
                    for field in ('colpali_page_rank','colpali_page_score','colpali_chunk_rank',
                                  'colpali_point_id','visual_block_id','visual_node_id',
                                  'image_path','page_image_path','bbox','visual_retrieval_text'):
                        if field in record:
                            item[field] = record[field]
                item['retrieval_streams'].append(name)
        return sorted(fused.values(), key=lambda r: r['rrf_score'], reverse=True)

    @staticmethod
    def rerank(query, candidates, top_k=8, max_context_tokens=3000, **kwargs):
        ranked = lexical_search(query, candidates, limit=top_k)
        output = []
        remaining = max_context_tokens
        for item in ranked:
            # This is not a neural score: the generation gate must remain closed.
            item.update(rerank_score=0, fusion_method='lexical_only_degraded')
            size = len(item.get('text', '').split()) * 1.35 + 12
            if size <= remaining:
                output.append(item)
                remaining -= size
        return output


def cached_inference(namespace, identity):
    from django.core.cache import cache
    try:
        return cache.get(_inference_key(namespace, identity))
    except Exception:
        return None


def cache_inference(namespace, identity, value, timeout=1800):
    from django.core.cache import cache
    try:
        cache.set(_inference_key(namespace, identity), value, timeout)
    except Exception:
        pass  # A cache outage must not prevent real inference.


def _inference_key(namespace, identity):
    return 'inference:v1:' + namespace + ':' + hashlib.sha256(
        json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def rank_indexed_records(terms, rows, count, average, limit=60):
    """BM25 over authorized postings supplied by a repository; no database access."""
    import math
    from collections import Counter
    frequencies=Counter(t for _,counts,_ in rows for t in terms if t in counts)
    ranked=[]
    for record,counts,length in rows:
        score=sum(math.log(1+(count-frequencies[t]+.5)/(frequencies[t]+.5))
            *counts[t]*2.5/(counts[t]+1.5*(.25+.75*length/average)) for t in terms if counts.get(t))
        if score>0:ranked.append(dict(record,lexical_score=round(score,6),score=score))
    return sorted(ranked,key=lambda r:r['lexical_score'],reverse=True)[:limit]
