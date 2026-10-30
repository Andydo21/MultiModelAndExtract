"""Collect real 2–3 page arXiv PDFs, then launch the main extraction runner."""
import concurrent.futures
import hashlib
import json
import os
import sys
import time
import threading
import xml.etree.ElementTree as ET
from pathlib import Path

import pymupdf
import requests

ROOT = Path('/app')
DEST = ROOT / 'PDF_180_2_3_TRANG'
MANIFEST = DEST / 'manifest.json'
SOURCES = ROOT / 'NGUON_180_TAI_LIEU_2_3_TRANG.md'
GROUPS = ['cs.RO', 'cs.CV', 'physics.app-ph', 'physics.ao-ph', 'astro-ph.EP',
          'physics.med-ph', 'q-bio.QM', 'q-bio.BM', 'q-fin.ST', 'econ.GN',
          'math.OC', 'stat.ML', 'cs.LG', 'physics.bio-ph', 'physics.soc-ph']
NS = {'a': 'http://www.w3.org/2005/Atom'}
PDF_LOCK = threading.Lock()


def save(items):
    DEST.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST.with_suffix('.part')
    tmp.write_text(json.dumps({'total': 180, 'completed': len(items), 'items': items},
                              ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, MANIFEST)
    lines = ['# 180 tài liệu đa lĩnh vực, 2–3 trang', '',
             'Số trang được kiểm tra trực tiếp trên PDF tải về.', '']
    lines += [f"{x['number']}. [{x['title']}]({x['url']}) — {x['pages']} trang; {x['category']}"
              for x in items]
    SOURCES.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def metadata(group):
    params = {'search_query': f'(co:"2 pages" OR co:"3 pages") AND cat:{group}',
              'start': 0, 'max_results': 200, 'sortBy': 'submittedDate', 'sortOrder': 'descending'}
    for attempt in range(5):
        try:
            response = requests.get('https://export.arxiv.org/api/query', params=params, timeout=90)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            result = []
            for entry in root.findall('a:entry', NS):
                identifier = entry.findtext('a:id', default='', namespaces=NS).rsplit('/', 1)[-1]
                title = ' '.join(entry.findtext('a:title', default='', namespaces=NS).split())
                if identifier and title:
                    result.append({'id': identifier, 'title': title, 'category': group,
                                   'url': 'https://arxiv.org/pdf/' + identifier})
            return result
        except (requests.RequestException, ET.ParseError) as error:
            print('metadata retry', group, attempt, str(error)[:160], flush=True)
            time.sleep(5 * (attempt + 1))
    return []


def fetch(item):
    for attempt in range(3):
        try:
            r = requests.get(item['url'], timeout=(20, 100))
            r.raise_for_status()
            data = r.content
            with PDF_LOCK:
                with pymupdf.open(stream=data, filetype='pdf') as pdf:
                    pages = len(pdf)
            if pages not in (2, 3):
                return None
            return item, data, pages, hashlib.sha256(data).hexdigest()
        except (requests.RequestException, RuntimeError, ValueError) as error:
            if attempt == 2:
                print('download failed', item['id'], str(error)[:120], flush=True)
            else:
                time.sleep(2 * (attempt + 1))
    return None


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    items = json.loads(MANIFEST.read_text(encoding='utf-8'))['items'] if MANIFEST.exists() else []
    ids = {x['id'].split('v')[0] for x in items}
    hashes = {x['sha256'] for x in items}
    pools = []
    for group in GROUPS:
        if len(items) >= 180:
            break
        candidates = metadata(group)
        pools.append(candidates)
        print('group', group, 'candidates', len(candidates), 'accepted so far', len(items), flush=True)
        time.sleep(3)
        accept(candidates, items, ids, hashes, min(15, 180 - len(items)))
    for candidates in pools:
        if len(items) >= 180:
            break
        accept(candidates, items, ids, hashes, 180 - len(items))
    save(items)
    if len(items) != 180:
        raise RuntimeError(f'Only found {len(items)} qualifying PDFs; extraction not started')
    print('180 valid PDFs ready; starting main extraction', flush=True)
    os.execv(sys.executable, [sys.executable, '/app/evaluation/run_180_pdf_main.py'])


def accept(candidates, items, ids, hashes, target):
    count = 0
    # Small batches bound memory and avoid downloading the entire search result.
    for offset in range(0, len(candidates), 6):
        if count >= target:
            break
        batch = [x for x in candidates[offset:offset + 6] if x['id'].split('v')[0] not in ids]
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            for found in pool.map(fetch, batch):
                if found is None or count >= target:
                    continue
                item, data, pages, digest = found
                base_id = item['id'].split('v')[0]
                if base_id in ids or digest in hashes:
                    continue
                number = len(items) + 1
                filename = f'{number:03d}_{item["id"].replace("/", "_")}.pdf'
                (DEST / filename).write_bytes(data)
                item.update(number=number, file=filename, pages=pages, sha256=digest,
                            size=len(data), status='downloaded')
                items.append(item)
                ids.add(base_id)
                hashes.add(digest)
                count += 1
                save(items)
    print('accepted', count, 'total', len(items), flush=True)


if __name__ == '__main__':
    main()
