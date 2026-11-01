from pathlib import Path
from importlib.util import find_spec
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import skipUnless
from unittest.mock import Mock, patch
import hashlib
import io
from django.test import SimpleTestCase, override_settings
from PIL import Image
from ai_extract.services.contract_pipeline_service import surya_page, remote_page
from ai_extract.services.layout_extractor import SuryaExtractor
from ai_extract.services.surya_inference_service import VISUAL_OCR_POLICY, BODY_OCR_POLICY
from ai_extract.repositories import page_inference_cache_repository as cache


class SuryaOCRTests(SimpleTestCase):
    def test_warns_when_ocr_cell_combines_distinct_source_columns(self):
        from ai_extract.services.surya_inference_service import table_structure_warnings
        sources = [dict(row=2, column=1, text='Model A'), dict(row=2, column=2, text='Model B')]
        merged = [dict(text='Model A\nModel B')]
        separate = [dict(text='Model A'), dict(text='Model B')]
        self.assertIn('multiple_source_columns_combined_in_single_cell', table_structure_warnings(merged,sources,2))
        self.assertNotIn('multiple_source_columns_combined_in_single_cell', table_structure_warnings(separate,sources,2))

    def test_table_comparison_tolerates_label_footnote_apostrophe_only(self):
        from ai_extract.services.surya_inference_service import normalize_table_ocr_comparison as normalize
        self.assertEqual(normalize("Robot model'1"), normalize('Robot model1'))
        self.assertNotEqual(normalize("10'2"), normalize('102'))
        self.assertNotEqual(normalize("can't"), normalize('cant'))
        self.assertNotEqual(normalize("Robot model'2"), normalize('Robot model1'))

    @skipUnless(find_spec('latex2mathml'), 'vision runtime provides latex2mathml')
    def test_table_comparison_accepts_tex_subscripts_without_losing_digits(self):
        from ai_extract.services.surya_inference_service import normalize_table_ocr_comparison as normalize
        source = 'Four circuits (\u03c64 x 2, \u03c66 x 2)'
        self.assertEqual(normalize(source), normalize(r'Four circuits (\phi_4 \times 2, \phi_6 \times 2)'))
        self.assertNotEqual(normalize(source), normalize(r'Four circuits (\phi_4 \times 2, \phi_5 \times 2)'))
        self.assertNotEqual(normalize('part_4'), normalize('part4'))

    def test_table_cell_keeps_high_accuracy_superscript(self):
        import numpy as np
        from ai_extract.services.surya_inference_service import _recognize_physical_table
        grid = dict(rows=1, columns=1, column_origin='model_columns', x_cuts=[0,100], y_cuts=[0,100], cells=[
            dict(row=1, column=1, rowspan=1, colspan=1, bbox=[0,0,100,100])])
        detailed = '0.2kgm²'
        predictor = SimpleNamespace(manager=SimpleNamespace(generate=Mock(return_value=[SimpleNamespace(raw=detailed, error=False)])))
        with patch('ai_extract.services.surya_inference_service.table_rule_observations', return_value=dict(ink=np.ones((100,100)))), \
             patch('ai_extract.services.surya_inference_service.table_crop_has_text', return_value=True), \
             patch('ai_extract.services.surya_inference_service.corroborated_table_grid', return_value=grid), \
             patch.dict('sys.modules', {'surya.inference.schema':SimpleNamespace(BatchInputItem=SimpleNamespace)}):
            result = _recognize_physical_table(Image.new('RGB',(100,100)), predictor,
                [dict(row=2,column=1,text='0.2kgm2')], column_predictor=lambda im:dict(columns=[[0,0,100,100]]), force_cell_ocr=True)
        self.assertEqual(result['table_rows'], [[detailed]])
        self.assertEqual(predictor.manager.generate.call_args.args[0][0].prompt_type, 'block')

    def test_new_cell_text_is_retained_with_warning_when_unverified(self):
        import numpy as np
        from ai_extract.services.surya_inference_service import _recognize_physical_table
        for second,accept in (('33 mm',True),('34 mm',False)):
            ink=np.zeros((200,100),dtype=np.uint8);ink[30:50,30:60]=255;ink[130:150,30:60]=255
            grid=dict(rows=2,columns=1,column_origin='model_columns',x_cuts=[0,100],y_cuts=[0,100,200],cells=[
                dict(row=r,column=1,rowspan=1,colspan=1,bbox=[0,(r-1)*100,100,r*100]) for r in (1,2)])
            def output(text):return SimpleNamespace(raw='<p>'+text+'</p>',error=False)
            predictor=SimpleNamespace(manager=SimpleNamespace(generate=Mock(side_effect=[
                [output('Label'),output('33 mm')],[output(second)],[output('33 mm')]])))
            with self.subTest(second=second), \
                 patch('ai_extract.services.surya_inference_service.table_rule_observations',return_value=dict(ink=ink)), \
                 patch('ai_extract.services.surya_inference_service.corroborated_table_grid',return_value=grid), \
                 patch.dict('sys.modules',{'surya.inference.schema':SimpleNamespace(BatchInputItem=SimpleNamespace)}):
                def recognize():return _recognize_physical_table(Image.new('RGB',(100,200)),predictor,[dict(row=1,column=1,text='Label')],
                    column_predictor=lambda im:dict(columns=[[0,0,100,200]]),force_cell_ocr=True)
                if accept:
                    result=recognize()
                    self.assertEqual(result['table_rows'],[['Label'],['33 mm']])
                    self.assertEqual(len(result['table_structure_review']['additional_cell_quotations']),1)
                else:
                    result=recognize()
                    self.assertEqual(result['table_rows'],[['Label'],['33 mm']])
                    self.assertIn('cell_ocr_differs_from_source_quotations',result['table_structure_review']['structure_warnings'])

    def test_subcolumn_boundary_can_move_between_rows(self):
        import numpy as np
        from ai_extract.services.surya_inference_service import corroborated_table_grid
        rules=dict(ink=np.ones((200,200)),horizontal=[[0,y,200,y] for y in (0,100,200)],
            vertical=[[80,0,80,100],[120,100,120,200]])
        prediction=dict(rows=[[0,0,200,100],[0,100,200,200]],columns=[[0,0,100,200],[100,0,200,200]],
            spans=[],structure_provider='surya_table_rec')
        grid=corroborated_table_grid(Image.new('RGB',(200,200)),rules,prediction,[dict(row=2,column=2,text='Value')])
        cells={(c['row'],c['column']):c for c in grid['cells']}
        self.assertEqual(cells[1,1]['bbox'][2],80)
        self.assertEqual(cells[1,2]['bbox'][0],80)
        self.assertEqual(cells[2,1]['bbox'][2],120)
        self.assertEqual(cells[2,2]['bbox'][0],120)

    def test_short_axis_separators_prevent_a_false_rowspan(self):
        import numpy as np
        from ai_extract.services.surya_inference_service import corroborated_table_grid
        image=Image.new('RGB',(200,300))
        rules=dict(ink=np.ones((300,200)),horizontal=[],horizontal_segments=[[100,100,200,100],[100,200,200,200]],
            vertical=[[100,0,100,300]])
        prediction=dict(rows=[[0,y,200,y+100] for y in (0,100,200)],columns=[[0,0,100,300],[100,0,200,300]],
            spans=[[100,0,200,300]],structure_provider='surya_table_rec')
        grid=corroborated_table_grid(image,rules,prediction,[dict(row=3,column=2,text='J3')])
        self.assertEqual(len(grid['cells']),4)
        self.assertEqual(grid['cells'][0]['rowspan'],3)
        self.assertEqual([(c['row'],c['rowspan']) for c in grid['cells'] if c['column']==2],[(1,1),(2,1),(3,1)])

    def test_cell_recovery_retains_disagreement_with_source_warning(self):
        import numpy as np
        from ai_extract.services.surya_inference_service import _recognize_physical_table
        ink=np.zeros((100,100),dtype=np.uint8);ink[30:60,30:65]=255
        grid=dict(rows=1,columns=1,column_origin='model_columns',x_cuts=[0,100],y_cuts=[0,100],cells=[
            dict(row=1,column=1,rowspan=1,colspan=1,bbox=[0,0,100,100])])
        predictor=SimpleNamespace(manager=SimpleNamespace(generate=Mock(return_value=[SimpleNamespace(raw='<p>33 mm</p>',error=False)])))
        # Deliberately disagreeing source dimensions force pixel OCR instead
        # of the source-quotation reuse path.
        with patch('ai_extract.services.surya_inference_service.table_rule_observations',return_value=dict(ink=ink)), \
             patch('ai_extract.services.surya_inference_service.corroborated_table_grid',return_value=grid), \
             patch.dict('sys.modules',{'surya.inference.schema':SimpleNamespace(BatchInputItem=SimpleNamespace)}):
            result=_recognize_physical_table(Image.new('RGB',(100,100)),predictor,[dict(row=2,column=1,text='32 mm')],
                column_predictor=lambda im:dict(columns=[[0,0,100,100]]),force_cell_ocr=True)
        self.assertEqual(result['table_rows'],[['33 mm']])
        self.assertEqual(result['table_structure_review']['missing_source_quotations'],['32 mm'])
        self.assertIn('source_quotations_not_covered_by_cell_ocr',result['table_structure_review']['structure_warnings'])

    def test_separator_only_crop_is_empty_but_isolated_I_glyph_is_not(self):
        import numpy as np
        from ai_extract.services.surya_inference_service import table_crop_has_text
        ink=np.zeros((100,100),dtype=np.uint8);ink[:,10]=255;ink[20,:]=255
        self.assertFalse(table_crop_has_text(ink))
        ink[35:70,50:53]=255
        self.assertTrue(table_crop_has_text(ink))
        ink[5:95,70:73]=255
        self.assertTrue(table_crop_has_text(ink[:,65:80]))
    def test_native_recovery_trims_pixel_empty_padding_not_content(self):
        import numpy as np
        from ai_extract.services.surya_inference_service import native_table_structure
        ink=np.zeros((100,200),dtype=np.uint8);ink[40:60,30:60]=255
        result=SimpleNamespace(error=False,rows=[SimpleNamespace(bbox=b) for b in ([0,0,200,20],[0,20,200,100])],
            cols=[SimpleNamespace(bbox=b) for b in ([0,0,100,100],[100,0,200,100])])
        with patch.dict('sys.modules',{'surya.table_rec':SimpleNamespace(TableRecPredictor=Mock(return_value=Mock(return_value=[result])))}), \
             patch('ai_extract.services.surya_inference_service.table_rule_observations',return_value=dict(
                 ink=ink,horizontal_mask=np.zeros_like(ink))):
            prediction=native_table_structure(Image.new('RGB',(200,100)),SimpleNamespace(manager=Mock()),
                dict(rows=[],columns=[[0,0,200,100]]),[dict(row=1,column=2,text='IP65')])
        self.assertEqual(prediction['rows'],[[0,20,200,100]])
        self.assertEqual(len(prediction['columns']),2)
        self.assertTrue(prediction['columns_confirmed'])

    def test_empty_ocr_cell_needs_second_prompt_and_preserves_all_source_text(self):
        import numpy as np
        from ai_extract.services.surya_inference_service import _recognize_physical_table
        grid=dict(rows=1,columns=2,column_origin='model_columns',x_cuts=[0,100,200],y_cuts=[0,100],cells=[
            dict(row=1,column=c,rowspan=1,colspan=1,bbox=[(c-1)*100,0,c*100,100]) for c in (1,2)])
        predictor=SimpleNamespace(manager=SimpleNamespace(generate=Mock(side_effect=[
            [SimpleNamespace(raw='<p>32 mm</p>',error=False),SimpleNamespace(raw='',error=False)],
            [SimpleNamespace(raw='<div></div>',error=False)]])))
        with patch('ai_extract.services.surya_inference_service.table_rule_observations',return_value=dict(ink=np.ones((100,200)))), \
             patch('ai_extract.services.surya_inference_service.corroborated_table_grid',return_value=grid), \
             patch.dict('sys.modules',{'surya.inference.schema':SimpleNamespace(BatchInputItem=SimpleNamespace)}):
            result=_recognize_physical_table(Image.new('RGB',(200,100)),predictor,
                [dict(row=1,column=1,text='32 mm')],column_predictor=lambda im:dict(columns=[[0,0,200,100]]))
        self.assertEqual(result['table_rows'],[['32 mm','']])
        self.assertEqual(result['table_cells'][1]['ocr_status'],'confirmed_empty_crop')
        calls = predictor.manager.generate.call_args_list
        self.assertEqual(calls[0].args[0][0].prompt_type, 'block')
        self.assertEqual(calls[1].args[0][0].prompt_type, 'block')

    def test_wrapped_body_is_read_from_physical_cell_not_shifted_model_row_centers(self):
        import numpy as np
        from ai_extract.services.surya_inference_service import recognize_physical_table
        source=[dict(row=r,column=1,rowspan=1,colspan=1,text=text,header=r<3)
                for r,text in enumerate(('Code','Header','1: Diagonal','ground','electrode'),1)]
        grid=dict(rows=3,columns=1,column_origin='model_columns',x_cuts=[0,200],y_cuts=[0,100,200,400],cells=[
            dict(row=r,column=1,rowspan=1,colspan=1,bbox=[0,a,200,b]) for r,a,b in ((1,0,100),(2,100,200),(3,200,400))])
        predictor=SimpleNamespace(manager=SimpleNamespace(generate=Mock(return_value=[
            SimpleNamespace(raw='<p>1: Diagonal<br>ground<br>electrode</p>',error=False)])))
        prediction=dict(columns=[[0,0,200,400]],rows=[[0,a,200,b] for a,b in ((0,100),(100,150),(150,190),(200,300),(300,400))])
        with patch('ai_extract.services.surya_inference_service.table_rule_observations',return_value=dict(ink=np.ones((400,200)))), \
             patch('ai_extract.services.surya_inference_service.corroborated_table_grid',return_value=grid), \
             patch.dict('sys.modules',{'surya.inference.schema':SimpleNamespace(BatchInputItem=SimpleNamespace)}):
            result=recognize_physical_table(Image.new('RGB',(200,400)),predictor,source,column_predictor=lambda im:prediction)
        self.assertEqual(result['table_rows'][1],['Header'])
        self.assertEqual(result['table_rows'][2],['1: Diagonal\nground\nelectrode'])
        self.assertTrue(result['table_cells'][1]['header'])
        self.assertEqual(result['table_cells'][2]['ocr_status'],'completed')
        self.assertEqual(len(predictor.manager.generate.call_args.args[0]),1)

    def test_overmerged_span_requires_whitespace_and_independent_cell_ocr(self):
        import numpy as np
        from ai_extract.services.surya_inference_service import recognize_physical_table
        ink=np.zeros((100,300),dtype=np.uint8)
        for x in (30,130,230):ink[30:50,x:x+20]=255
        source=[dict(row=1,column=c,text=t) for c,t in enumerate(('Label','0.28 sec','0.31 sec'),1)]
        grid=dict(rows=1,columns=3,column_origin='model_columns',x_cuts=[0,100,200,300],y_cuts=[0,100],cells=[
            dict(row=1,column=1,rowspan=1,colspan=1,bbox=[0,0,100,100]),
            dict(row=1,column=2,rowspan=1,colspan=2,bbox=[100,0,300,100])])
        predictor=SimpleNamespace(manager=SimpleNamespace(generate=Mock(return_value=[
            SimpleNamespace(raw='<p>0.28 sec</p>',error=False),SimpleNamespace(raw='<p>0.31 sec</p>',error=False)])))
        with patch('ai_extract.services.surya_inference_service.table_rule_observations',return_value=dict(ink=ink)), \
             patch('ai_extract.services.surya_inference_service.corroborated_table_grid',return_value=grid), \
             patch.dict('sys.modules',{'surya.inference.schema':SimpleNamespace(BatchInputItem=SimpleNamespace)}):
            result=recognize_physical_table(Image.new('RGB',(300,100)),predictor,source,
                column_predictor=lambda im:dict(columns=[[0,0,300,100]]))
        self.assertEqual(result['table_rows'],[['Label','0.28 sec','0.31 sec']])
        self.assertEqual(len(predictor.manager.generate.call_args.args[0]),2)

    def test_model_columns_cannot_be_replaced_by_glyph_strokes(self):
        import numpy as np
        from ai_extract.services.surya_inference_service import corroborated_table_grid
        image=Image.new('RGB',(400,200));rules=dict(ink=np.ones((200,400),dtype=np.uint8),
            horizontal=[[0,0,400,0],[0,100,400,100],[0,200,400,200]],
            vertical=[[x,20,x,60] for x in range(10,390,10)])
        prediction=dict(columns=[[0,0,200,200],[200,0,400,200]],rows=[[0,0,400,100],[0,100,400,200]],spans=[])
        cells=[dict(row=2,column=2,text='Value')]
        grid=corroborated_table_grid(image,rules,prediction,cells)
        self.assertEqual((grid['rows'],grid['columns'],len(grid['cells'])),(2,2,4))

    def test_tatr_rows_are_not_collapsed_by_missing_physical_rules(self):
        import numpy as np
        from ai_extract.services.surya_inference_service import corroborated_table_grid
        image=Image.new('RGB',(500,800));rules=dict(ink=np.ones((800,500),dtype=np.uint8),vertical=[],
            horizontal=[[0,y,500,y] for y in (0,100,200,800)])
        prediction=dict(columns=[[x,0,x+100,800] for x in range(0,500,100)],
            rows=[[0,y*800/14,500,(y+1)*800/14] for y in range(14)],spans=[])
        grid=corroborated_table_grid(image,rules,prediction,[dict(row=14,column=5,text='Value')])
        self.assertEqual((grid['rows'],grid['columns'],len(grid['cells'])),(14,5,70))
        self.assertEqual(grid['row_origin'],'tatr_model_rows')

    def test_partial_vertical_rules_do_not_merge_borderless_data_columns(self):
        import numpy as np
        from ai_extract.services.surya_inference_service import corroborated_table_grid
        image=Image.new('RGB',(400,200))
        rules=dict(ink=np.ones((200,400),dtype=np.uint8),
            horizontal=[[0,0,400,0],[0,100,400,100],[0,200,400,200]],
            vertical=[[200,0,200,200]])
        prediction=dict(columns=[[x,0,x+100,200] for x in (0,100,200,300)],
            rows=[[0,0,400,100],[0,100,400,200]],spans=[])
        grid=corroborated_table_grid(image,rules,prediction,[])
        self.assertEqual(len(grid['cells']),8)
        self.assertTrue(all(c['colspan']==1 and c['rowspan']==1 for c in grid['cells']))

    def test_independent_column_disagreement_withholds_structure(self):
        import numpy as np
        from ai_extract.services.surya_inference_service import corroborated_table_grid
        with self.assertRaisesRegex(ValueError,'column predictions disagree'):
            corroborated_table_grid(Image.new('RGB',(200,100)),dict(ink=np.ones((100,200)),horizontal=[],vertical=[]),
                dict(columns=[[0,0,200,100]],rows=[],spans=[]),[dict(row=1,column=2,text='Two columns')])

    def test_unrelated_vertical_stroke_below_closed_grid_is_not_a_clipped_border(self):
        from ai_extract.services.surya_inference_service import physical_table_grid
        grid=physical_table_grid(600,330,[[0,0,600,0],[0,150,600,150],[0,300,600,300]],
            [[0,0,0,300],[300,0,300,300],[600,100,600,330]])
        self.assertEqual((grid['rows'],grid['columns']),(2,2))

    def test_large_aligned_table_reuses_source_ocr_without_spending_cell_ocr_budget(self):
        import numpy as np
        from ai_extract.services.surya_inference_service import recognize_physical_table
        source=[dict(row=r+1,column=c+1,rowspan=1,colspan=1,text=f'{r}:{c}',header=r==0)
                for r in range(17) for c in range(10)]
        cells=[dict(c,bbox=[(c['column']-1)*30,(c['row']-1)*30,c['column']*30,c['row']*30]) for c in source]
        ink=np.full((510,300),255,dtype=np.uint8)
        rules=dict(horizontal=[],vertical=[],ink=ink,horizontal_mask=np.zeros_like(ink),vertical_mask=np.zeros_like(ink))
        predictor=SimpleNamespace(manager=SimpleNamespace(generate=Mock(return_value=[])))
        with patch('ai_extract.services.surya_inference_service.table_rule_observations',return_value=rules), \
             patch('ai_extract.services.surya_inference_service.physical_table_grid',return_value=dict(
                 rows=17,columns=10,cells=cells,column_origin='model_columns',x_cuts=[],y_cuts=[])), \
             patch.dict('sys.modules',{'surya.inference.schema':SimpleNamespace(BatchInputItem=SimpleNamespace)}):
            result=recognize_physical_table(Image.new('RGB',(300,510)),predictor,source,
                column_predictor=lambda im:[[0,0,300,510]])
        self.assertEqual(len(result['table_cells']),170)
        self.assertTrue(all(c['ocr_status']=='source_ocr_order_aligned' for c in result['table_cells']))
        predictor.manager.generate.assert_called_once_with([])

    def test_double_vertical_rules_are_one_physical_column_boundary(self):
        from ai_extract.services.surya_inference_service import physical_table_grid
        grid=physical_table_grid(2000,500,[[0,0,2000,0],[0,250,2000,250],[0,500,2000,500]],
            [[x,0,x,500] for x in [0,496,504,1000,1496,1504,1995]])
        self.assertEqual((grid['rows'],grid['columns']),(2,4))
        self.assertEqual(grid['x_cuts'],[0,500,1000,1500,2000])

    def test_partial_rule_refines_approximate_model_column_for_rowspan(self):
        from ai_extract.services.surya_inference_service import physical_table_grid
        grid=physical_table_grid(600,300,[[0,0,600,0],[190,150,600,150],[0,300,600,300]],[],
            [[0,0,195,300],[215,0,400,300],[400,0,600,300]])
        self.assertEqual((grid['rows'],grid['columns']),(2,3))
        self.assertEqual(grid['x_cuts'][1],190)
        self.assertEqual(grid['cells'][0]['rowspan'],2)

    def test_physical_empty_cell_rejects_displaced_source_text_and_border_rim(self):
        import numpy as np
        from ai_extract.services.surya_inference_service import recognize_physical_table
        ink=np.zeros((100,200),dtype=np.uint8)
        ink[10:80,2:4]=255  # A separator remnant at the cropped left edge.
        ink[35:50,140:160]=255
        predictor=SimpleNamespace(manager=SimpleNamespace(generate=Mock(return_value=[])))
        rules=dict(horizontal=[[0,0,200,0],[0,100,200,100]],vertical=[[100,0,100,100]],
                   ink=ink,horizontal_mask=np.zeros_like(ink),vertical_mask=np.zeros_like(ink))
        cells=[dict(row=1,column=1,text='Correct title',header=False),
               dict(row=1,column=2,text='Correct title',header=False)]
        with patch('ai_extract.services.surya_inference_service.table_rule_observations',return_value=rules), \
             patch.dict('sys.modules',{'surya.inference.schema':SimpleNamespace(BatchInputItem=SimpleNamespace)}):
            result=recognize_physical_table(Image.new('RGB',(200,100)),predictor,cells,
                column_predictor=lambda im:[[0,0,100,100],[100,0,200,100]])
        self.assertEqual(result['table_rows'],[['','Correct title']])
        self.assertEqual(result['table_cells'][0]['ocr_status'],'empty_source_crop')
        predictor.manager.generate.assert_called_once_with([])

    def test_physical_rows_do_not_depend_on_text_line_spacing(self):
        from ai_extract.services.surya_inference_service import physical_table_grid
        grid=physical_table_grid(300,1200,[[0,0,300,0],[0,1200,300,1200]],
            [[0,0,0,1200],[300,0,300,1200]],[[0,0,150,1200],[150,0,300,1200]])
        self.assertEqual((grid['rows'],grid['columns']),(1,2))
        self.assertEqual(len(grid['cells']),2)

    def test_partial_rule_retains_a_merged_cell(self):
        from ai_extract.services.surya_inference_service import physical_table_grid
        grid=physical_table_grid(300,200,[[0,0,300,0],[100,100,300,100],[0,200,300,200]],
            [[0,0,0,200],[100,0,100,200],[200,0,200,200],[300,0,300,200]])
        cell=next(c for c in grid['cells'] if c['row']==1 and c['column']==1)
        self.assertEqual(cell['rowspan'],2)
        self.assertEqual(len(grid['cells']),5)

    def test_clipped_bottom_cannot_silently_discard_body_text(self):
        from ai_extract.services.surya_inference_service import physical_table_grid
        with self.assertRaisesRegex(ValueError,'bottom is clipped'):
            physical_table_grid(300,400,[[0,0,300,0],[0,100,300,100]],
                [[0,0,0,390],[150,0,150,390],[300,0,300,390]])

    def test_missing_row_rules_require_review(self):
        from ai_extract.services.surya_inference_service import physical_table_grid
        with self.assertRaisesRegex(ValueError,'row separators'):
            physical_table_grid(300,400,[],[],[[0,0,150,400],[150,0,300,400]])

    def test_table_crop_decoder_requires_one_complete_grid(self):
        from ai_extract.services.surya_inference_service import source_html_metadata
        for html in ('<table><tr><td>A',
            '<table><tr><td>A</td></tr></table><table><tr><td>B</td></tr></table>',
            '<table><tr><td>A</td></tr><tr><td>B</td><td>C</td></tr></table>'):
            data=source_html_metadata(html,strict_table=True)
            self.assertEqual(data['table_structure_status'],'unresolved')
            self.assertNotIn('table_cells',data)
        valid=source_html_metadata('<table><tr><td colspan="2"></td><td>Right</td></tr>'
            '<tr><td>A</td><td>B</td><td>C</td></tr></table>',strict_table=True)
        self.assertEqual(valid['table_cells'][1]['column'],3)

    def test_cell_paragraphs_and_line_breaks_remain_in_one_cell(self):
        from ai_extract.services.surya_inference_service import source_html_metadata
        data=source_html_metadata('<table><tr><td><p>First line</p><p>Second line<br>Third line</p></td><td>Value</td></tr></table>')
        self.assertEqual(len(data['table_rows']),1)
        self.assertEqual(len(data['table_cells']),2)
        self.assertEqual(data['table_cells'][0]['text'].splitlines(),['First line','','Second line','Third line'])

    def test_empty_cells_and_thead_roles_are_preserved(self):
        from ai_extract.services.surya_inference_service import source_html_metadata
        data=source_html_metadata('<table><thead><tr><td colspan="2"></td><td>Name</td></tr></thead>'
            '<tbody><tr><td>A</td><td></td><td>B</td></tr></tbody></table>')
        self.assertEqual(data['table_cells'][1]['column'],3)
        self.assertTrue(data['table_cells'][1]['header'])
        self.assertFalse(data['table_cells'][2]['header'])
        self.assertEqual(data['table_cells'][3]['text'],'')

    def test_invalid_or_overlapping_spans_cannot_publish_partial_grid(self):
        from ai_extract.services.surya_inference_service import source_html_metadata
        for html in ('<table><tr><td rowspan="0">Bad</td></tr></table>',
            '<table><tr><td>A</td><td rowspan="2">B</td></tr><tr><td colspan="2">C</td></tr></table>'):
            data=source_html_metadata(html)
            self.assertEqual(data['table_structure_status'],'unresolved')
            self.assertNotIn('table_cells',data)

    def test_force_ocr_bypasses_both_disk_caches_but_reuses_current_page_in_memory(self):
        from ai_extract.services.contract_pipeline_service import _cached_page
        extractor=SimpleNamespace(force_ocr=True)
        infer=Mock(return_value=dict(layout=[],ocr_lines=[]))
        with patch.object(cache,'read') as read,patch.object(cache,'write') as write:
            first=_cached_page(extractor,'same-page','current','_page',infer,
                legacy_key='legacy',upgrade=Mock())
            second=_cached_page(extractor,'same-page','current','_page',infer,
                legacy_key='legacy',upgrade=Mock())
        infer.assert_called_once()
        read.assert_not_called();write.assert_not_called()
        self.assertIs(first,second)
        self.assertFalse(first['cache_hit'])
        self.assertFalse(first['body_cache_hit'])
        self.assertTrue(first['forced_fresh_ocr'])

    def test_rejected_ocr_cannot_publish_table_cells_from_discarded_html(self):
        from ai_extract.services.document_graph_service import append_layout_edges
        nodes=[]
        leaf=dict(chunk_id='bad',text='',bbox=[10,10,100,100],layout_type='table',
            structure_metadata=dict(ocr_status='unresolved',table_rows=[['Invented']],
                table_cells=[dict(row=1,column=1,text='Invented')]))
        append_layout_edges(dict(leaves=[leaf]),lambda key,**props:nodes.append(props),lambda *args,**props:None)
        self.assertFalse(any(n['category'] in {'table','table_row','table_cell'} for n in nodes))

    def test_picture_text_is_retained_without_second_ocr_pass(self):
        from ai_extract.services.surya_inference_service import SuryaStreamManager,surya_visual_page
        for first_text in ('HSR', 'A A A A A A A A A A'):
            manager=SuryaStreamManager(Mock())
            manager.generate=Mock()
            child=self.block(0,'<p>'+first_text+'</p>')
            predictor=Mock(manager=manager,return_value=[SimpleNamespace(blocks=[child])])
            with patch('ai_extract.services.surya_inference_service._visual_layout',return_value=[SimpleNamespace(error=False,bboxes=[])]),patch.dict(
                    'sys.modules',{'surya.inference.schema':SimpleNamespace(BatchInputItem=SimpleNamespace)}):
                result=surya_visual_page(Image.new('RGB',(400,400)),predictor,
                    [dict(label='Picture',bbox=[0,0,400,400],region_id='figure')])
            root=result['regions'][0]
            self.assertEqual([line['text'] for line in root['ocr_lines']],[first_text])
            self.assertEqual(root['children'][0]['ocr_status'],'completed')
            self.assertEqual(root['children'][0]['ocr_validation'],'surya_single_pass_unverified')
            manager.generate.assert_not_called()

    def test_native_html_retains_heading_and_merged_table_cells(self):
        from ai_extract.services.surya_inference_service import source_html_metadata
        self.assertEqual(source_html_metadata('<h3>Robot specifications</h3>')['heading_level'],3)
        metadata=source_html_metadata('<table><tr><th rowspan="2">Item</th><th colspan="2">Models</th></tr>'
            '<tr><td>A</td><td>B</td></tr><tr><td>Load</td><td colspan="2">8 kg</td></tr></table>')
        cells=metadata['table_cells']
        self.assertEqual(cells[0]['rowspan'],2)
        self.assertEqual(cells[2]['column'],2)
        self.assertEqual(cells[-1]['colspan'],2)
        self.assertEqual(metadata['table_rows'][-1],['Load','8 kg',''])
        self.assertEqual(metadata['table_header_origin'],'surya_html_unreviewed')
        self.assertNotIn('heading_level',source_html_metadata('<p>5.1 Any text</p>'))

    def test_old_body_cache_cannot_skip_improved_ocr(self):
        from ai_extract.services.contract_pipeline_service import _cached_page
        old_key=hashlib.sha256(b'old-body').hexdigest();new_key=hashlib.sha256(b'new-body').hexdigest()
        cache.write(old_key,dict(layout=[],ocr_lines=[],metrics={}))
        infer=Mock(return_value=dict(layout=[],ocr_lines=[],cacheable=True))
        upgrade=Mock()
        _cached_page(SimpleNamespace(),'image',new_key,'_page',infer,legacy_key=old_key,upgrade=upgrade)
        infer.assert_called_once();upgrade.assert_not_called()

    def test_visual_text_duplicate_requires_matching_source_position(self):
        from ai_extract.services.surya_inference_service import merge_visual_ocr
        import copy
        page=dict(layout=[dict(label='Text',bbox=[10,10,100,30],region_id='text'),
                          dict(label='Picture',bbox=[0,0,200,200],region_id='image')],
                  ocr_lines=[dict(text='EVP2',bbox=[10,10,100,30],region_id='text')])
        def supplement(box):
            return dict(regions=[dict(index=1,bbox=[0,0,200,200],label='Picture',status='completed',error_type='',
                children=[dict(region_id='image/0',parent_region_id='image',label='Text',bbox=box)],
                ocr_lines=[dict(region_id='image/0',parent_region_id='image',text='EVP2',bbox=box)])],
                metrics=dict(elapsed_s=1),cacheable=True)
        duplicate=merge_visual_ocr(copy.deepcopy(page),supplement([10,10,100,30]))
        self.assertEqual(len(duplicate['ocr_lines']),1)
        self.assertEqual(len(duplicate['layout']),2)
        elsewhere=merge_visual_ocr(copy.deepcopy(page),supplement([10,100,100,120]))
        self.assertEqual(len(elsewhere['ocr_lines']),2)
        self.assertEqual(len(elsewhere['layout']),3)

    def test_failed_visual_text_cannot_enter_lexical_retrieval(self):
        from ai_extract.services.document_structure_service import build_document_structure
        from shared.retrieval import lexical_search
        tree=build_document_structure([dict(chunk_id='bad',text='Bangladesh invented text',layout_type='figure',
            bbox=[0,0,100,100],page_number=1,structure_metadata={'visual_ocr_status':'failed'})])
        self.assertTrue(tree['leaves'][0]['retrieval_excluded'])
        self.assertEqual(lexical_search('Bangladesh',tree['leaves']),[])

    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        override = override_settings(A3_PAGE_CACHE_ROOT=Path(directory.name))
        override.enable()
        self.addCleanup(override.disable)

    def block(self, order, html, **extra):
        return SimpleNamespace(reading_order=order, html=html, bbox=[10,20,200,100],
            label='Text', confidence=.88, skipped=False, error=False, **extra)

    def test_html_preserves_contract_lines_table_cells_and_reading_order(self):
        predictor = Mock(return_value=[SimpleNamespace(blocks=[
            self.block(1, '<table><tr><td>300</td><td>USD</td></tr><tr><td>15</td><td>days</td></tr></table>'),
            self.block(0, '<h2>Article 1</h2><p>Payment &amp; delivery</p>')])])
        image = Image.new('RGB', (400,400))
        page = surya_page(image, predictor)
        predictor.assert_called_once_with([image], full_page=True)
        self.assertEqual(page['model'], 'surya-ocr-v2')
        self.assertEqual(page['ocr_lines'][0]['text'], 'Article 1\nPayment & delivery')
        self.assertEqual(page['ocr_lines'][1]['text'], '300\tUSD\n15\tdays')
        self.assertEqual(page['ocr_lines'][1]['bbox'], [10,20,200,100])

    def test_failed_block_keeps_bbox_but_never_partial_text_or_success_cache(self):
        block = self.block(0, '')
        block.error = True
        page=surya_page(Image.new('RGB',(20,20)), Mock(return_value=[SimpleNamespace(blocks=[block])]))
        self.assertEqual(page['ocr_lines'],[])
        self.assertEqual(page['layout'][0]['ocr_status'],'failed')
        self.assertFalse(page['cacheable'])

    def visual_prediction(self, raw, error=False, child_box=None, sharp=None):
        body=self.block(0,'<p>Body source</p>')
        body.bbox=[10,250,200,300]
        visual=self.block(1,''); visual.label='Picture'; visual.skipped=True
        child=self.block(0,raw);child.bbox=child_box or [0,0,190,80];child.error=error
        predictor=Mock(manager=SimpleNamespace(),side_effect=[
            [SimpleNamespace(blocks=[body,visual])],[SimpleNamespace(blocks=[child])]])
        with patch('ai_extract.services.surya_inference_service._visual_layout',return_value=[SimpleNamespace(error=False)]):
            page=surya_page(Image.new('RGB',(400,400)),predictor,sharp)
        return page,predictor

    def test_visual_crop_masks_page_text_and_rejects_clipped_duplicate(self):
        from ai_extract.services.surya_inference_service import surya_visual_page
        page=Image.new('RGB',(140,100),'black')
        sharp=Image.new('RGB',(280,200),'black')
        duplicate=self.block(0,'<p>clipped page text</p>');duplicate.bbox=[5,2,18,16]
        inside=self.block(1,'<p>Inside image</p>');inside.bbox=[70,40,120,60]
        predictor=Mock(manager=SimpleNamespace(),return_value=[SimpleNamespace(blocks=[duplicate,inside])])
        def layout(crops,manager):
            self.assertEqual(crops[0].getpixel((10,10)),(255,255,255))
            self.assertEqual(crops[0].getpixel((100,100)),(0,0,0))
            return [SimpleNamespace(error=False)]
        with patch('ai_extract.services.surya_inference_service._visual_layout',side_effect=layout):
            result=surya_visual_page(page,predictor,[
                dict(label='Text',bbox=[0,0,60,20],region_id='text'),
                dict(label='Picture',bbox=[40,0,140,100],region_id='image')],high_resolution_image=sharp)
        self.assertEqual([l['text'] for l in result['regions'][0]['ocr_lines']],['Inside image'])
        self.assertEqual(result['regions'][0]['excluded_page_text'],1)
        self.assertEqual(result['regions'][0]['status'],'completed')
        self.assertEqual(sharp.getpixel((100,10)),(0,0,0))

    def test_skipped_picture_is_read_without_changing_parent_layout_or_bbox(self):
        page,predictor=self.visual_prediction('<p>AX-203</p><p>24 V</p>')
        self.assertEqual(page['layout'][1]['label'],'Picture')
        self.assertEqual(page['layout'][1]['bbox'],[10,20,200,100])
        self.assertEqual(page['layout'][2]['parent_region_id'],'surya:1')
        self.assertEqual(page['ocr_lines'][-1]['region_id'],'surya:1/0')
        self.assertEqual(page['ocr_lines'][-1]['text'],'AX-203\n24 V')
        self.assertEqual(page['ocr_lines'][-1]['bbox'],[10,20,200,100])
        self.assertEqual(page['ocr_lines'][-1]['text_bbox_scope'],'surya_layout_region')
        self.assertFalse(predictor.call_args.kwargs['full_page'])

    def test_visual_model_output_is_retained_as_unverified(self):
        page,_=self.visual_prediction('<img alt="A diagram"/><p>The diagram illustrates a system.</p>')
        self.assertTrue(page['cacheable'])
        self.assertEqual(page['layout'][1]['visual_ocr_status'],'completed')
        self.assertEqual([l['text'] for l in page['ocr_lines']],['Body source','The diagram illustrates a system.'])
        self.assertEqual(page['layout'][-1]['ocr_validation'],'surya_single_pass_unverified')

    def test_child_ocr_maps_sharper_pixels_to_original_page_coordinates(self):
        page,_=self.visual_prediction('<p>AX-203</p>',child_box=[6,8,40,32],sharp=Image.new('RGB',(800,800)))
        self.assertEqual(page['ocr_lines'][-1]['text'],'AX-203')
        self.assertEqual(page['ocr_lines'][-1]['bbox'],[13,24,30,36])
        self.assertTrue(page['cacheable'])

    def test_visual_failure_retains_body_without_accepting_partial_text(self):
        page,_=self.visual_prediction('<p>unfinished',error=True)
        self.assertEqual([l['text'] for l in page['ocr_lines']],['Body source'])
        self.assertFalse(page['cacheable'])
        self.assertEqual(page['layout'][1]['visual_ocr_status'],'failed')

    def test_text_label_is_not_dropped_by_arbitrary_font_height(self):
        page,_=self.visual_prediction('<p>EVP2</p>',child_box=[3,4,45,10])
        self.assertEqual(page['ocr_lines'][-1]['text'],'EVP2')
        self.assertEqual(page['layout'][-1]['ocr_status'],'completed')

    def test_picture_without_text_remains_successful_empty_visual(self):
        page,_=self.visual_prediction('<p></p>')
        self.assertTrue(page['cacheable'])
        self.assertEqual(page['layout'][1]['visual_ocr_status'],'no_text')
        self.assertEqual(len(page['ocr_lines']),1)

    def test_failed_child_does_not_erase_readable_sibling(self):
        from ai_extract.services.surya_inference_service import surya_visual_page
        good=self.block(0,'<p>EVP2 Runtime</p>');good.bbox=[5,5,195,30]
        bad=self.block(1,'<p>unfinished');bad.bbox=[5,80,195,105];bad.error=True
        predictor=Mock(manager=SimpleNamespace(),return_value=[SimpleNamespace(blocks=[good,bad])])
        with patch('ai_extract.services.surya_inference_service._visual_layout',return_value=[SimpleNamespace(error=False)]):
            result=surya_visual_page(Image.new('RGB',(400,400)),predictor,
                [dict(label='Picture',bbox=[0,0,200,150])])
        row=result['regions'][0]
        self.assertEqual(row['status'],'partial')
        self.assertEqual(row['text'],'')  # Parent is an image, not a text chunk.
        self.assertEqual(row['ocr_lines'][0]['text'],'EVP2 Runtime')
        self.assertEqual(len(row['ocr_lines']),1)
        self.assertFalse(result['cacheable'])

    def test_inner_picture_is_ignored_without_deeper_layout_or_ocr(self):
        from ai_extract.services.surya_inference_service import surya_visual_page
        nested=self.block(0,'');nested.label='Picture';nested.skipped=True;nested.bbox=[10,20,110,120]
        child=self.block(0,'<p>Camera</p>');child.bbox=[5,8,80,30]
        predictor=Mock(manager=SimpleNamespace(),side_effect=[
            [SimpleNamespace(blocks=[nested])],[SimpleNamespace(blocks=[child])]])
        with patch('ai_extract.services.surya_inference_service._visual_layout',side_effect=[
                [SimpleNamespace(error=False)],[SimpleNamespace(error=False)]]) as detect:
            result=surya_visual_page(Image.new('RGB',(400,400)),predictor,
                [dict(label='Picture',bbox=[100,100,300,300])])
        self.assertEqual(detect.call_count,1)
        row=result['regions'][0]
        self.assertEqual(row['ocr_lines'],[])
        self.assertEqual(row['children'],[])
        self.assertEqual(row['status'],'no_text')
        predictor.assert_called_once()

    def test_identical_picture_layout_stops_instead_of_recursing_forever(self):
        from ai_extract.services.surya_inference_service import surya_visual_page
        nested=self.block(0,'');nested.label='Picture';nested.skipped=True;nested.bbox=[0,0,200,200]
        predictor=Mock(manager=SimpleNamespace(),return_value=[SimpleNamespace(blocks=[nested])])
        with patch('ai_extract.services.surya_inference_service._visual_layout',return_value=[SimpleNamespace(error=False)]) as detect:
            result=surya_visual_page(Image.new('RGB',(400,400)),predictor,
                [dict(label='Picture',bbox=[100,100,300,300])])
        self.assertEqual(detect.call_count,1)
        self.assertEqual(result['regions'][0]['status'],'no_text')

    def test_small_image_frame_preserves_pixels_and_remaps_layout_boxes(self):
        from ai_extract.services.surya_inference_service import _visual_layout
        inside=SimpleNamespace(bbox=[210,235,280,265],polygon=[])
        outside=SimpleNamespace(bbox=[0,0,50,50],polygon=[])
        layout=SimpleNamespace(bboxes=[inside,outside],image_bbox=[0,0,512,512])
        def detect(frames):
            self.assertEqual(frames[0].getpixel((206,231)),(0,0,0))
            self.assertEqual(frames[0].getpixel((0,0)),(255,255,255))
            return [layout]
        with patch.dict('sys.modules',{'surya.layout':SimpleNamespace(LayoutPredictor=Mock(return_value=detect))}):
            results=_visual_layout([Image.new('RGB',(100,50),'black')],SimpleNamespace())
        self.assertEqual(results[0].bboxes,[inside])
        self.assertEqual(inside.polygon,[[4,4],[74,4],[74,34],[4,34]])
        self.assertEqual(results[0].image_bbox,[0,0,100,50])

    def test_small_picture_labels_are_sent_to_ocr(self):
        from ai_extract.services.surya_inference_service import surya_visual_page
        tiny=SimpleNamespace(label='Text',bbox=[0,0,25,3])
        readable=SimpleNamespace(label='Text',bbox=[0,10,160,28])
        layout=SimpleNamespace(error=False,bboxes=[tiny,readable])
        block=self.block(0,'<p>Readable label</p>');block.bbox=readable.bbox
        def recognize(images,layout_results,full_page):
            self.assertEqual(layout_results[0].bboxes,[tiny,readable])
            return [SimpleNamespace(blocks=[block])]
        predictor=Mock(manager=SimpleNamespace(),side_effect=recognize)
        with patch('ai_extract.services.surya_inference_service._visual_layout',return_value=[layout]):
            result=surya_visual_page(Image.new('RGB',(1241,1755)),predictor,
                [dict(label='Picture',bbox=[0,0,500,500],region_id='image')])
        self.assertEqual(result['metrics']['ignored_tiny_text_regions'],0)
        self.assertEqual(result['metrics']['failed'],0)
        self.assertEqual(result['metrics']['unreadable_regions'],0)
        self.assertEqual(result['regions'][0]['ocr_lines'][0]['text'],'Readable label')

    def test_failed_image_layout_is_retried_once_without_repeating_healthy_siblings(self):
        from ai_extract.services.surya_inference_service import _visual_layout
        for retry_failed in (False,True):
            good=SimpleNamespace(error=False,bboxes=[],image_bbox=[0,0,300,200])
            bad=SimpleNamespace(error=True,bboxes=[],image_bbox=[0,0,300,200])
            box=SimpleNamespace(bbox=[42,52,82,72],polygon=[])
            rescued=SimpleNamespace(error=retry_failed,bboxes=[box],image_bbox=[0,0,364,264])
            detect=Mock(side_effect=[[good,bad],[rescued]])
            with self.subTest(retry_failed=retry_failed), patch.dict('sys.modules',{'surya.layout':SimpleNamespace(LayoutPredictor=Mock(return_value=detect))}):
                result=_visual_layout([Image.new('RGB',(300,200),'black') for _ in range(2)],SimpleNamespace())
            self.assertEqual(detect.call_count,2)
            self.assertEqual(len(detect.call_args_list[1].args[0]),1)
            self.assertIs(result[0],good)
            self.assertEqual(result[1].error,retry_failed)
            self.assertEqual(box.polygon,[[10,20],[50,20],[50,40],[10,40]])

    def test_extractor_keeps_one_chunk_per_region_without_bbox_text_leak(self):
        extractor=SuryaExtractor.__new__(SuryaExtractor)
        regions=[SimpleNamespace(region_id='surya:0',label='Picture',bbox=[0,0,300,300],chunk_policy='surya_layout_region'),
                 SimpleNamespace(region_id='surya:0/0',parent_region_id='surya:0',label='Text',bbox=[10,10,100,100],chunk_policy='surya_layout_region'),
                 SimpleNamespace(region_id='surya:0/1',parent_region_id='surya:0',label='Text',bbox=[10,10,100,100],chunk_policy='surya_layout_region')]
        text='First source\n'+('Long original source '*150)
        extractor._ocr_png_image=Mock(return_value=[dict(region_id='surya:0/0',text=text,bbox=[10,10,100,100]),
            dict(region_id='surya:0/1',text='Different source',bbox=[10,10,100,100])])
        extractor.surya_layout_model=Mock(return_value=[SimpleNamespace(bboxes=regions)])
        extractor._draw_visual_bbox_crop=Mock(return_value='/media/figure.png')
        chunks,count=extractor._extract_chunks_from_page_image(Image.new('RGB',(400,400)),1,1,'test',Path('page.png'),'/media/page.png')
        self.assertEqual(count,4)
        # The image has an explicit type label; its child OCR stays separate.
        self.assertIn('Hình ảnh kỹ thuật',chunks[0]['text'])
        self.assertNotIn('First source',chunks[0]['text'])
        self.assertEqual([c['text'] for c in chunks[1:]],[text,'Different source'])
        self.assertEqual(chunks[1]['surya_parent_region_id'],'surya:0')

    def test_unbound_ocr_is_preserved_without_guessing_paragraph_groups(self):
        extractor=SuryaExtractor.__new__(SuryaExtractor)
        extractor.surya_layout_model=None
        extractor._ocr_png_image=Mock(return_value=[
            dict(region_id='missing',text='First source',bbox=[0,0,100,20]),
            dict(text='Adjacent different source',bbox=[0,20,100,40])])
        chunks,count=extractor._extract_chunks_from_page_image(
            Image.new('RGB',(200,200)),1,1,'test',Path('page.png'),'/media/page.png')
        self.assertEqual([c['text'] for c in chunks],['First source','Adjacent different source'])
        self.assertEqual(count,3)
        self.assertTrue(all(c['reading_order_unresolved'] for c in chunks))
        extractor._ocr_png_image.assert_called_once()

    def test_atomic_chunks_keep_multiline_text_and_image_scope_in_graph(self):
        from ai_extract.services.document_structure_service import build_document_structure
        text='1. Label inside image\n'+('Original source '*100)
        def region(key,parent,kind,content):
            return dict(chunk_id=key,text=content,layout_type=kind,page_number=1,image_path='page.png',bbox=[10,10,500,500],
                structure_metadata=dict(chunk_policy='surya_layout_region',surya_region_id=key,surya_parent_region_id=parent,
                                        reading_order=0,ocr_status='completed'))
        records=[region('heading',None,'title','Document heading'),region('figure',None,'figure',''),
                 region('inside','figure','title',text),region('after',None,'paragraph','Outside image')]
        tree=build_document_structure(records,child_chars=64)
        self.assertEqual(len(tree['leaves']),4)
        inside=next(l for l in tree['leaves'] if l['source_chunk_id']=='inside')
        self.assertEqual(inside['text'],text)
        nodes={n['id']:n for n in tree['nodes']}
        self.assertEqual(nodes[nodes[inside['node_id']]['parent_id']]['kind'],'figure')
        after=next(l for l in tree['leaves'] if l['source_chunk_id']=='after')
        self.assertEqual(nodes[after['node_id']]['title'],'Document heading')

    @override_settings(RAG_INFERENCE_URL='http://inference.test')
    def test_legacy_cache_reads_only_visual_crops_and_reuses_one_page_response(self):
        image=Image.new('RGB',(400,400))
        buffer=io.BytesIO(); image.save(buffer,format='PNG')
        digest=hashlib.sha256(buffer.getvalue()).hexdigest()
        old_key=hashlib.sha256(('surya-v2-cache1:http://inference.test:'+digest).encode()).hexdigest()
        layout=[dict(label='Picture',bbox=[10,20,200,100])]
        cache.write(old_key,dict(model='surya-ocr-v2',layout=layout,metrics=dict(body_ocr_policy=BODY_OCR_POLICY),ocr_lines=[dict(text='Body source',bbox=[0,200,300,300])]))
        response=Mock()
        response.json.return_value=dict(regions=[dict(index=0,label='Picture',bbox=[10,20,200,100],text='AX-203',
            status='completed',error_type='',confidence=.9,
            children=[dict(region_id='surya:0/0',parent_region_id='surya:0',label='Text',bbox=[10,20,200,100],confidence=.9,chunk_policy='surya_layout_region')],
            ocr_lines=[dict(text='AX-203',bbox=[10,20,200,100],region_id='surya:0/0')])],cacheable=True,
            metrics=dict(policy=VISUAL_OCR_POLICY,elapsed_s=.1,regions=1,failed=0))
        extractor=SimpleNamespace()
        with patch('ai_extract.services.contract_pipeline_service.requests.post',return_value=response) as post:
            result=remote_page(extractor,image)
            self.assertEqual([l['text'] for l in result['ocr_lines']],['Body source','AX-203'])
            self.assertTrue(result['body_cache_hit'])
            self.assertTrue(result['visual_cache_upgraded'])
            self.assertTrue(post.call_args.args[0].endswith('/rag/ocr_visual_regions'))
            remote_page(extractor,image)
            self.assertEqual(post.call_count,1)

    @override_settings(RAG_INFERENCE_URL='http://inference.test')
    def test_failed_cache_upgrade_keeps_body_and_retries_next_extraction(self):
        import requests
        image=Image.new('RGB',(400,400)); buffer=io.BytesIO(); image.save(buffer,format='PNG')
        digest=hashlib.sha256(buffer.getvalue()).hexdigest()
        cache.write(hashlib.sha256(('surya-v2-cache1:http://inference.test:'+digest).encode()).hexdigest(),
            dict(model='surya-ocr-v2',layout=[dict(label='Picture',bbox=[10,20,200,100])],
                 metrics=dict(body_ocr_policy=BODY_OCR_POLICY),
                 ocr_lines=[dict(text='Body source',bbox=[0,200,300,300])]))
        with patch('ai_extract.services.contract_pipeline_service.requests.post',side_effect=requests.ReadTimeout()) as post:
            result=remote_page(SimpleNamespace(),image)
            self.assertEqual(result['ocr_lines'][0]['text'],'Body source')
            self.assertFalse(result['cacheable'])
            remote_page(SimpleNamespace(),image)
            self.assertEqual(post.call_count,2)

    @override_settings(RAG_INFERENCE_URL='')
    def test_local_layout_and_ocr_share_cached_prediction(self):
        predictor = Mock(return_value=[SimpleNamespace(blocks=[self.block(0,'<p>Payment</p>')])])
        with TemporaryDirectory() as directory, override_settings(MEDIA_ROOT=Path(directory)), \
             patch.object(SuryaExtractor, '_shared_ocr_predictor', None), \
             patch('ai_extract.services.surya_inference_service.create_surya_predictor', return_value=predictor):
            extractor = SuryaExtractor()
            image = Image.new('RGB',(400,400))
            lines = extractor._ocr_png_image(image)
            layout = extractor.surya_layout_model([image])[0]
            self.assertEqual(lines[0]['text'], 'Payment')
            self.assertEqual(layout.bboxes[0].bbox, lines[0]['bbox'])
            predictor.assert_called_once()

    @override_settings(RAG_INFERENCE_URL='http://inference.test')
    def test_remote_requires_new_surya_protocol(self):
        response = Mock()
        response.json.return_value = dict(model='legacy-ocr', layout=[], ocr_lines=[])
        with patch('ai_extract.services.contract_pipeline_service.requests.post', return_value=response):
            with self.assertRaises(ValueError):
                remote_page(SimpleNamespace(), Image.new('RGB',(20,20)))
