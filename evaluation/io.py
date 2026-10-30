"""Artifact IO. Raw records are appended, hash chained and fsynced."""
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def sha_file(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def new_output(base, label):
    path = Path(base) / (label + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: canonical(v) if isinstance(v, (dict, list)) else v for k, v in row.items() if k in fields})


def read_records(path):
    rows, previous = [], None
    with Path(path).open(encoding='utf-8') as stream:
        for index, line in enumerate(stream):
            record = json.loads(line)
            content = {k:v for k,v in record.items() if k != 'record_sha256'}
            if (record.get('sequence') != index or record.get('previous_sha256') != previous
                    or digest(content) != record.get('record_sha256')):
                raise ValueError(f'Raw log integrity failure at record {index}')
            rows.append(record)
            previous = record['record_sha256']
    return rows


def append_record(path, value):
    path = Path(path)
    rows = read_records(path) if path.exists() else []
    record = dict(value, sequence=len(rows), previous_sha256=rows[-1]['record_sha256'] if rows else None)
    record['record_sha256'] = digest(record)
    # Each experiment owns a unique directory; do not share this writer across workers.
    with path.open('a', encoding='utf-8') as stream:
        stream.write(canonical(record) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    return record
