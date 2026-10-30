import importlib.util
from unittest import skipUnless
from django.test import SimpleTestCase
from shared.retrieval import lexical_search, query_policy, gate_evidence, check_citation_ids, evidence_identity, LexicalOnlyReranker

class RetrievalPolicyTests(SimpleTestCase):

    def test_unproven_graph_paths_do_not_open_generation_gate(self):
        self.assertFalse(gate_evidence('motor', [dict(text='motor', layout_type='neo4j_graph_node', rerank_score=99)])['accepted'])

    def test_exact_codes_do_not_match_substrings(self):
        records = [dict(text='E-404 motor', original_name='log'), dict(text='E-4040 motor', original_name='log')]
        self.assertEqual(len(lexical_search('E-404', records)), 1)

    def test_corpus_not_limited_to_first_records(self):
        records = [dict(text='unrelated', original_name='log') for _ in range(450)]
        records.append(dict(text='E-404 overheating', original_name='target'))
        self.assertEqual(lexical_search('E-404', records)[0]['original_name'], 'target')

    def test_identifier_policy(self):
        self.assertEqual(query_policy('E-404')['intent'], 'exact_identifier')
        self.assertEqual(query_policy('E-404')['weights'], query_policy('quy trình')['weights'])
        self.assertEqual(set(query_policy('hình vẽ')['weights'].values()), {1.0})

    def test_unknown_identifier_closes_gate(self):
        citations = [dict(text='E-405 motor', rerank_score=99)]
        self.assertFalse(gate_evidence('E-404', citations)['accepted'])

    def test_empty_and_weak_evidence_closes_gate(self):
        self.assertFalse(gate_evidence('motor', [])['accepted'])
        self.assertFalse(gate_evidence('motor', [dict(text='motor', rerank_score=12)])['accepted'])

    def test_citation_identifiers(self):
        citations = [dict(evidence_id='E1')]
        self.assertTrue(check_citation_ids('Nguồn [E1]', citations)['valid'])
        self.assertFalse(check_citation_ids('Nguồn [E2]', citations)['valid'])
        self.assertFalse(check_citation_ids('Không có nguồn', citations)['valid'])

    def test_identity_preserves_zero_and_distinct_graph_paths(self):
        self.assertTrue(evidence_identity(dict(document_id=1, chunk_id=0)).endswith(':0'))
        self.assertNotEqual(evidence_identity(dict(text='path1')), evidence_identity(dict(text='path2')))

    def test_stream_duplicates_do_not_inflate_fusion(self):
        item = dict(document_id=1, chunk_id=0, text='motor')
        fused = LexicalOnlyReranker.reciprocal_rank_fusion({'dense': [item, item], 'keyword': [item]})
        self.assertEqual(len(fused), 1)
        self.assertAlmostEqual(fused[0]['rrf_score'], 2 / 61)

    def test_degraded_mode_does_not_claim_neural_score(self):
        result = LexicalOnlyReranker.rerank('motor', [dict(text='motor')])
        self.assertEqual(result[0]['rerank_score'], 0)
        self.assertFalse(gate_evidence('motor', result)['accepted'])

class OfficeParserTests(SimpleTestCase):

    def test_pptx_uses_presentation_slide_order(self):
        import tempfile
        from pathlib import Path
        from zipfile import ZipFile
        from ai_extract.services.office_parser import extract_pptx
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'deck.pptx'
            with ZipFile(path, 'w') as archive:
                archive.writestr('ppt/presentation.xml', '<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><p:sldIdLst><p:sldId r:id="r2"/><p:sldId r:id="r1"/></p:sldIdLst></p:presentation>')
                archive.writestr('ppt/_rels/presentation.xml.rels', '<Relationships><Relationship Id="r1" Target="slides/slide1.xml"/><Relationship Id="r2" Target="slides/slide2.xml"/></Relationships>')
                for number in (1, 2):
                    archive.writestr(f'ppt/slides/slide{number}.xml', f'<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Slide content {number}</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>')
            chunks = extract_pptx(path)
            self.assertEqual(chunks[0]['text'], 'Slide content 2')
            self.assertEqual(chunks[0]['source_location'], 'Slide 1')
            self.assertFalse(chunks[0]['has_image'])

    @skipUnless(importlib.util.find_spec('openpyxl'), 'openpyxl is required for XLSX parsing')
    def test_xlsx_preserves_merged_headers_cell_addresses_and_formulas(self):
        import tempfile
        from pathlib import Path
        from openpyxl import Workbook
        from ai_extract.services.office_parser import extract_spreadsheet
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'spec.xlsx'
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = 'Specs'
            sheet.append(['Voltage', None, 'Current'])
            sheet.merge_cells('A1:B1')
            sheet.append([24, 12, '=A2+B2'])
            workbook.save(path)
            workbook.close()
            chunks = extract_spreadsheet(path)
            row = chunks[1]
            self.assertEqual(row['table_headers'][:2], ['Voltage', 'Voltage'])
            self.assertEqual(row['cell_values']['A2'], '24')
            self.assertEqual(row['merged_ranges'], ['A1:B1'])
            self.assertIn('chưa có giá trị cache', row['cell_values']['C2'])
            self.assertEqual(row['formulas']['C2'], '=A2+B2')

    def test_tsv_and_multi_letter_columns(self):
        import tempfile
        from pathlib import Path
        from ai_extract.services.office_parser import extract_spreadsheet, column_name
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.tsv'
            path.write_text('Part\tValue\nCN3\t24 V\n', encoding='utf-8')
            self.assertEqual(extract_spreadsheet(path)[1]['cell_values']['B2'], '24 V')
        self.assertEqual(column_name(27), 'AA')
