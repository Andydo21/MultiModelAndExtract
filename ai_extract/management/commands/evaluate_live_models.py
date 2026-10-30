"""Small, labeled synthetic evaluation using existing live model endpoints."""
import base64
import io
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import requests
from django.conf import settings
from django.core.management.base import BaseCommand
from ai_extract.services.evaluation_metrics_service import ocr_counts, ranking_metrics, field_metrics
from shared.retrieval import lexical_search, LexicalOnlyReranker
from ai_extract.services.contract_pipeline_service import HybridBackend, request_qwen


PASSAGES = [
    ('payment', 'Bên A thanh toán 120.000.000 đồng trong vòng 30 ngày kể từ ngày nhận đủ hóa đơn và biên bản nghiệm thu.'),
    ('safety', 'Trước khi bảo trì tủ điện, kỹ thuật viên phải ngắt nguồn điện và khóa cầu dao.'),
    ('duration', 'Hợp đồng có hiệu lực 12 tháng kể từ ngày ký.'),
    ('motor', 'Động cơ M1 được cấp nguồn qua cầu dao QF1. Điện áp định mức là 380 V.'),
    ('temperature', 'Nhiệt độ vận hành cho phép của cảm biến PT100 là từ -20 đến 80 độ C.'),
    ('bearing', 'Kiểm tra vòng bi và bổ sung dầu bôi trơn sau mỗi 6 tháng.'),
    ('delivery', 'Bên bán giao hàng tại kho Hà Nội trong 10 ngày kể từ ngày nhận đơn đặt hàng.'),
    ('warranty', 'Thiết bị được bảo hành 24 tháng. Lỗi do vận hành sai không thuộc phạm vi bảo hành.'),
    ('force', 'Sự kiện bất khả kháng phải được thông báo trong vòng 5 ngày.'),
    ('torque', 'Siết bu lông M8 với mô men 25 Nm. Sử dụng cờ lê lực đã hiệu chuẩn.'),
    ('invoice', 'Hồ sơ đề nghị thanh toán gồm hóa đơn hợp lệ và biên bản nghiệm thu có chữ ký.'),
    ('lockout', 'Cô lập nguồn năng lượng và áp dụng khóa an toàn trước khi sửa chữa thiết bị điện.'),
]
QUERIES = [
    ('pay', 'Bao lâu phải trả tiền và cần chứng từ nào?', ['payment', 'invoice']),
    ('safe', 'Cần cô lập năng lượng như thế nào trước khi sửa tủ điện?', ['safety', 'lockout']),
    ('voltage', 'M1 dùng điện áp bao nhiêu và nối qua thiết bị nào?', ['motor']),
    ('heat', 'PT100 chịu được dải nhiệt độ nào?', ['temperature']),
    ('tighten', 'Lực siết bu lông M8 là bao nhiêu?', ['torque']),
    ('cover', 'Nhà cung cấp bảo hành thiết bị trong bao lâu?', ['warranty']),
]
FIELDS = ['COMPANY_NAME', 'TAX_CODE', 'CONTRACT_VALUE', 'DATE_EFFECTIVE', 'DATE_EXPIRE',
          'DURATION', 'PAYMENT_TERM', 'PENALTY', 'OBLIGATION']


class Command(BaseCommand):
    help = 'Measure real OCR/text retrieval/entities on small synthetic labeled fixtures; no database writes.'

    def add_arguments(self, parser):
        parser.add_argument('--output', default='contracts/tests/results/model_metrics_latest.json')
        parser.add_argument('--web-results', default='scratch/live_pdf_flow_results.json')

    def handle(self, **options):
        started = time.perf_counter()
        report = dict(created_at=datetime.now(ZoneInfo('Asia/Bangkok')).isoformat(),
                      scope='synthetic_smoke_not_representative_benchmark', real_model_responses=True,
                      normalization='Unicode NFC + collapse whitespace; case, accents and punctuation retained',
                      errors=[], classification={'status': 'not_measured', 'reason': 'No labeled document classifier in this evaluation.'},
                      numeric_prediction={'status': 'not_applicable', 'reason': 'Pipeline extracts numbers; it does not implement numeric regression.'})
        output = Path(options['output']); output.parent.mkdir(parents=True, exist_ok=True)
        def save():
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        for stage, operation in [('ocr', self.ocr), ('retrieval', self.retrieval), ('entities', self.entities)]:
            self.stdout.write(f'Start {stage}', ending='\n'); self.stdout.flush()
            try:
                report[stage] = operation()
                self.stdout.write(json.dumps({stage: report[stage].get('summary')}, ensure_ascii=False))
            except Exception as error:
                report[stage] = dict(status='failed', error_type=type(error).__name__)
                report['errors'].append(dict(stage=stage, error_type=type(error).__name__))
            save()
        path = Path(options['web_results'])
        if path.exists():
            web = json.loads(path.read_text(encoding='utf-8'))
            report['web_qa'] = self.web_qa(web, str(path))
        else:
            report['web_qa'] = dict(status='not_measured', reason='No web-flow results file.')
        report['elapsed_seconds'] = round(time.perf_counter() - started, 2)
        save()
        self.stdout.write(f'Report: {output}')

    def ocr(self):
        from PIL import Image, ImageDraw, ImageFont, ImageFilter
        fixtures = [
            ('contract_vi', ['Điều 1. Thanh toán', 'Bên A thanh toán 120.000.000 đồng trong vòng 30 ngày.',
                             'Hồ sơ gồm hóa đơn và biên bản nghiệm thu.'], False),
            ('technical_vi', ['Sơ đồ tủ điện', 'Cầu dao QF1 cấp nguồn cho động cơ M1.',
                              'Điện áp: 380 V. Mô men siết M8: 25 Nm.'], False),
            ('technical_en_scan', ['Maintenance procedure', 'Disconnect power before servicing.',
                                   'Motor M1: 380 V. Bolt M8: 25 Nm.'], True),
        ]
        rows = []
        font = ImageFont.truetype('/app/scratch/arial.ttf', 32)
        for identifier, lines, scan in fixtures:
            image = Image.new('RGB', (1200, 420), 'white'); draw = ImageDraw.Draw(image)
            for i, line in enumerate(lines):
                draw.text((45, 45 + i * 95), line, font=font, fill='black')
            if scan:
                image = image.rotate(1, fillcolor='white').filter(ImageFilter.GaussianBlur(.3))
            buffer = io.BytesIO(); image.save(buffer, format='PNG')
            before = time.perf_counter()
            response = requests.post(settings.RAG_INFERENCE_URL.rstrip('/') + '/rag/extract_page',
                                     json={'image_base64': base64.b64encode(buffer.getvalue()).decode()}, timeout=120)
            response.raise_for_status()
            prediction = '\n'.join(line['text'] for line in response.json()['ocr_lines'])
            counts = ocr_counts('\n'.join(lines), prediction)
            rows.append(dict(id=identifier, reference='\n'.join(lines), prediction=prediction,
                             seconds=round(time.perf_counter() - before, 3), **counts))
        chars = sum(r['reference_characters'] for r in rows); words = sum(r['reference_words'] for r in rows)
        return dict(status='measured', samples=rows, summary=dict(samples=len(rows),
                    cer=sum(r['character_errors'] for r in rows) / chars,
                    wer=sum(r['word_errors'] for r in rows) / words,
                    reference_characters=chars, reference_words=words))

    def retrieval(self):
        backend = HybridBackend()
        records = [dict(document_id=key, chunk_id=key, text=text, original_name='', heading_path=[]) for key, text in PASSAGES]
        # One embedding load followed by one reranker load; no repeated model switching.
        vectors = backend.embedding([r['text'] for r in records] + [q[1] for q in QUERIES])
        rows = []
        for i, (identifier, query, relevant) in enumerate(QUERIES):
            before = time.perf_counter(); q = vectors[len(records) + i]
            dense = sorted([dict(r, similarity=sum(a * b for a, b in zip(vector, q)))
                            for r, vector in zip(records, vectors[:len(records)])], key=lambda r: r['similarity'], reverse=True)
            lexical = lexical_search(query, records, limit=len(records))
            fused = LexicalOnlyReranker.reciprocal_rank_fusion({'keyword': lexical, 'dense': dense}, weights={'keyword': 1, 'dense': 1})
            reranked = backend.rerank(query, fused)
            rankings = {name: [r['document_id'] for r in ranked] for name, ranked in
                        [('bm25', lexical), ('bge', dense), ('rrf_rerank', reranked)]}
            scores = {name: {str(k): ranking_metrics(ids, relevant, k) for k in (1, 3, 5)} for name, ids in rankings.items()}
            rows.append(dict(id=identifier, query=query, relevant_ids=relevant, rankings=rankings,
                             metrics=scores, reranking_seconds=round(time.perf_counter() - before, 3)))
        summary = {name: {str(k): {metric: sum(row['metrics'][name][str(k)][metric] for row in rows) / len(rows)
                                  for metric in ('recall', 'reciprocal_rank', 'ndcg')} for k in (1, 3, 5)}
                   for name in ('bm25', 'bge', 'rrf_rerank')}
        return dict(status='measured', corpus_size=len(records), queries=len(rows), corpus=records,
                    samples=rows, summary=summary,
                    scope='text_only_BM25_BGE_equal_RRF_reranker; no_Qdrant_ColPali_graph_or_context_expansion',
                    relevance='manually_defined_binary_document_ids; no_weight_tuning_or_train_test_split')

    def entities(self):
        fixtures = [
            ('payment', PASSAGES[0][1], dict(CONTRACT_VALUE='120.000.000', DURATION='30 ngày',
                         PAYMENT_TERM='30 ngày kể từ ngày nhận đủ hóa đơn và biên bản nghiệm thu')),
            ('duration', PASSAGES[2][1], dict(DURATION='12 tháng kể từ ngày ký')),
        ]
        rows = []
        for identifier, text, gold in fixtures:
            expected = {key: gold.get(key, '') for key in FIELDS}
            prediction = request_qwen('extract_entities', {'text': text})['entities']
            before = field_metrics(expected, prediction)
            rows.append(dict(id=identifier, source=text, **before))
        tp, fp, fn = (sum(row[key] for row in rows) for key in ('tp', 'fp', 'fn'))
        positive_rows = [r for row in rows for r in row['rows'] if r['expected']]
        summary = dict(samples=len(rows), positive_fields=len(positive_rows),
                       positive_field_exact_match=sum(r['exact_match'] for r in positive_rows) / len(positive_rows),
                       precision=tp / (tp + fp) if tp + fp else None,
                       recall=tp / (tp + fn) if tp + fn else None,
                       f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                       tp=tp, fp=fp, fn=fn)
        return dict(status='measured', samples=rows, summary=summary,
                    scope='strict_value_exact_match; empty fields excluded from positive EM, hallucinated extra values count FP')

    def web_qa(self, web, path):
        rows = web.get('questions', [])
        if len(rows) != 4:
            return dict(status='not_measured', reason='Expected four questions from the synthetic PDF harness.', source_file=path)
        # Explicit fixture rubric; not an LLM-generated correctness score.
        labels = [('payment', True, ('30', 'hóa đơn', 'nghiệm thu')),
                  ('maintenance', True, ('ngắt', 'khóa')),
                  ('penalty_absent', False, ()), ('diagram_left', True, ('QF1', 'trái', 'M1'))]
        assessed = []
        for row, (identifier, answerable, values) in zip(rows, labels):
            generated = row.get('generation_status') == 'generated'
            passes = generated and all(v.casefold() in (row.get('answer') or '').casefold() for v in values)
            assessed.append(dict(id=identifier, answerable=answerable, generated=generated,
                                 fixture_values_present=passes if answerable else None,
                                 answer=row.get('answer'), diagnostics=row.get('generation_diagnostics'),
                                 retrieval_ms=row.get('coverage', {}).get('retrieval_latency_ms')))
        answerable = [r for r in assessed if r['answerable']]
        return dict(status='historical_web_fixture', source_file=path, samples=assessed,
                    summary=dict(questions=len(assessed), answerable=len(answerable),
                                 generated=sum(r['generated'] for r in assessed),
                                 complete_generated_answers=sum(r['fixture_values_present'] is True for r in answerable),
                                 answerable_completion_rate=sum(r['fixture_values_present'] is True for r in answerable) / len(answerable),
                                 unsupported_blocked=sum('model_rejected_unsupported_claim' in r['diagnostics'].get('failures', []) for r in assessed)),
                    faithfulness='not_measured_as_general_rate',
                    completeness='required fixture values only; no full human semantic audit',
                    structure=web['extract']['result'], frontend_job=web.get('frontend_job'), pages=web.get('pages'))
