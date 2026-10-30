"""Private, content-addressed cache of successful page inference responses."""
import json
import os
import tempfile
from pathlib import Path
from django.conf import settings


def _path(key):
    if len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
        raise ValueError('Invalid page cache key')
    return Path(getattr(settings, 'A3_PAGE_CACHE_ROOT', settings.BASE_DIR / 'scratch/a3-page-cache')) / (key + '.json')


def read(key):
    try:
        value = json.loads(_path(key).read_text(encoding='utf-8'))
        if value.get('model') == 'surya-ocr-v2' and isinstance(value.get('ocr_lines'), list) and isinstance(value.get('layout'), list):
            return value
    except (OSError, ValueError, AttributeError):
        pass
    return None


def write(key, value):
    target = _path(key)
    temporary = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=target.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False)
        os.replace(temporary, target)
    except OSError:
        # A full/unwritable cache must not discard successful OCR evidence.
        pass
    finally:
        if temporary and temporary.exists():
            temporary.unlink(missing_ok=True)
