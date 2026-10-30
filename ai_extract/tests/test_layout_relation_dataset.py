from django.test import SimpleTestCase
from ai_extract.management.commands.export_layout_relation_dataset import validate_page

class LayoutRelationDatasetTests(SimpleTestCase):
    def page(self):
        return dict(blocks=[dict(id='b1',bbox=[0,0,200,100])],regions=[dict(id='i1',bbox=[300,100,800,900])],
                    pairs=[dict(text_id='b1',region_id='i1',label=None,reviewer='')])
    def test_proximity_does_not_supply_training_label(self):
        p=self.page();validate_page(p)
        with self.assertRaises(ValueError):validate_page(p,True)
        p['pairs'][0].update(label=1,reviewer='reviewer');validate_page(p,True)
    def test_invalid_regions_and_source_ids_fail(self):
        p=self.page();p['regions'][0]['bbox']=[800,100,300,900]
        with self.assertRaises(ValueError):validate_page(p)
        p=self.page();p['pairs'][0]['region_id']='missing'
        with self.assertRaises(ValueError):validate_page(p)
