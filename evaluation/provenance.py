"""Actual source/corpus hashes and explicitly separated local/remote hardware."""
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .io import digest, sha_file, write_csv, write_json


def git(root, *args):
    try:
        return subprocess.check_output(['git',*args],cwd=root,text=True,encoding='utf-8',stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):
        return None


def source_hashes(root):
    files = {p for folder in ('evaluation','NewIdea/services','shared','ai_extract/services')
             for p in (root/folder).rglob('*.py')}
    command = root/'contracts/management/commands/run_experiment.py'
    if command.exists():
        files.add(command)
    return {p.relative_to(root).as_posix():sha_file(p) for p in sorted(files)}


def local_hardware():
    result = dict(platform=platform.platform(),cpu=platform.processor(),gpu=None,
                  scope='local preparation host; not the remote inference server')
    try:
        gpu = subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version',
                              '--format=csv,noheader'],capture_output=True,text=True,timeout=5)
        if gpu.returncode==0:
            result['gpu'] = gpu.stdout.strip().splitlines()
    except (OSError,subprocess.TimeoutExpired):
        pass
    return result


def freeze(root, config, out):
    import pymupdf
    root, out = Path(root), Path(out)
    inventory, pages, regions, seen_hashes = [], [], [], set()
    for path in sorted((root/config['corpus_dir']).rglob('*')):
        if path.suffix.lower()!='.pdf':
            continue
        sha = sha_file(path)
        did = 'doc-'+sha[:20]
        entry = dict(path=path.relative_to(root).as_posix(),document_id=did,sha256=sha,
                     bytes=path.stat().st_size,pages=None,error=None,duplicate_bytes=sha in seen_hashes)
        try:
            with pymupdf.open(path) as pdf:
                entry['pages']=len(pdf)
                if sha not in seen_hashes:
                    for pi,page in enumerate(pdf,1):
                        blocks=[b for b in page.get_text('blocks') if b[6]==0 and b[4].strip()]
                        pages.append(dict(document_id=did,page=pi,text_regions=len(blocks),
                            status='text_layer_only_unreviewed' if blocks else 'needs_ocr'))
                        for bi,b in enumerate(blocks):
                            regions.append(dict(evidence_id=f'{did}:p{pi}:b{bi}',document_id=did,
                                version=sha,page=pi,text=b[4].strip(),
                                region=dict(bbox=list(b[:4]),coordinate_system='PDF_points_top_left'),
                                breadcrumb=[path.stem,f'Page {pi}'],scope=f'{did}:p{pi}',links=[],
                                review_status='unreviewed',source='PDF_text_layer_not_production_OCR'))
        except Exception as error:
            entry['error']=type(error).__name__
        seen_hashes.add(sha)
        inventory.append(entry)
    if not inventory:
        raise ValueError('No actual PDFs found; refusing empty corpus snapshot')
    hashes=source_hashes(root)
    snapshot=dict(schema_version=1,created_at=datetime.now(timezone.utc).isoformat(),
        git_commit=git(root,'rev-parse','HEAD'),git_status=git(root,'status','--porcelain'),
        source_hashes=hashes,source_tree_sha256=digest(hashes),corpus=inventory,
        corpus_sha256=digest([(e['path'],e['sha256']) for e in inventory]),
        hardware_local=local_hardware(),hardware_inference=config['generator'].get('hardware'),
        generator=config['generator'],retrieval=config['retrieval'],config_sha256=digest(config))
    snapshot['snapshot_id']=digest(snapshot)
    write_json(out/'snapshot.json',snapshot)
    write_json(out/'evidence.pdf_text.json',regions)
    write_csv(out/'corpus.csv',inventory)
    write_csv(out/'pages.csv',pages)
    write_json(out/'annotation_queue.json',[dict(evidence_id=e['evidence_id'],document_id=e['document_id'],
        page=e['page'],region=e['region'],source_text=e['text'],question=None,reference_answer=None,
        ground_truth_tuples=None,reviewer=None,review_status='unannotated') for e in regions])
    return snapshot


def verify_snapshot(root,snapshot):
    content={k:v for k,v in snapshot.items() if k!='snapshot_id'}
    if digest(content)!=snapshot.get('snapshot_id'):
        raise ValueError('Snapshot manifest hash mismatch')
    for entry in snapshot['corpus']:
        path=Path(root)/entry['path']
        if not path.exists() or sha_file(path)!=entry['sha256']:
            raise ValueError('Corpus changed: '+entry['path'])
    if source_hashes(Path(root))!=snapshot['source_hashes']:
        raise ValueError('Evaluator/production source tree changed; create a new snapshot')
