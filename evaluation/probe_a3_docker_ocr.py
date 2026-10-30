"""Smoke-test the Docker GPU runtime against a real page from the corpus."""
import base64
import json
from pathlib import Path

import pymupdf
import requests

root = Path('/app/PDF_180_2_3_TRANG')
first = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))['items'][0]
with pymupdf.open(root / first['file']) as pdf:
    image = pdf[0].get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False).tobytes('png')
result = requests.post('http://a3-inference:8010/rag/extract_page',
                       json={'image_base64': base64.b64encode(image).decode('ascii')},
                       timeout=(10, 900))
result.raise_for_status()
data = result.json()
if not data.get('ocr_lines') or data.get('metrics', {}).get('failed_regions'):
    raise RuntimeError('Surya returned no OCR lines or failed regions')
print('Real-page OCR passed:', first['file'], 'lines:', len(data['ocr_lines']))
