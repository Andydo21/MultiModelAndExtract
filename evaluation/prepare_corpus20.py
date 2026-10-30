"""Stage actual selected PDFs and extract source pages; never invent source text."""
import json
import shutil
from pathlib import Path

import pymupdf

from .io import sha_file


def prepare(root=None):
    root=Path(root or Path(__file__).resolve().parents[1])
    base=root/'evaluation/benchmarks'
    selection=json.loads((base/'corpus20_selection.json').read_text(encoding='utf-8'))
    out=base/'corpus20'
    (out/'pdfs').mkdir(parents=True,exist_ok=True)
    pages,documents=[],[]
    for alias,name in selection['documents'].items():
        source=root/name
        target=out/'pdfs'/f'{alias}.pdf'
        if target.exists() and sha_file(target)!=sha_file(source):
            raise ValueError('Existing staged PDF differs: '+alias)
        if not target.exists():
            shutil.copy2(source,target)
        sha=sha_file(source)
        with pymupdf.open(source) as pdf:
            documents.append(dict(alias=alias,document_id='doc-'+sha[:20],sha256=sha,path=name,
                                  staged_path=target.relative_to(root).as_posix(),pages=len(pdf)))
            for pi,page in enumerate(pdf,1):
                pages.append(dict(alias=alias,page=pi,text=page.get_text(),
                                  width=page.rect.width,height=page.rect.height))
    (out/'manifest.json').write_text(json.dumps(dict(documents=documents),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'source_pages.json').write_text(json.dumps(pages,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Staged {len(documents)} actual PDFs; extracted {len(pages)} pages.')


if __name__=='__main__':
    prepare()
