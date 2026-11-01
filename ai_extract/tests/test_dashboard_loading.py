"""Dashboard query bounds; fixtures are isolated software test records."""
from django.test import TestCase
from contracts.models import Company, Contract, ContractVersion, User
from ai_extract.models import LayoutExtractionRun
from ai_extract.repositories.layout_repository import dashboard_contracts, latest_run


class DashboardLoadingTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.company = Company.objects.create(company_name='Dashboard test fixture', tax_code='fixture-only')
        cls.user = User.objects.create(username='dashboard-fixture', company=cls.company)
        cls.expected = {}
        for index in range(3):
            contract = Contract.objects.create(company=cls.company, contract_code=f'DASH-FIXTURE-{index}', title='Fixture')
            older = ContractVersion.objects.create(contract=contract, version_number=1)
            newest = ContractVersion.objects.create(contract=contract, version_number=2)
            LayoutExtractionRun.objects.create(version=older, status='COMPLETED', structure={'fixture': 'old'})
            LayoutExtractionRun.objects.create(version=newest, status='COMPLETED', structure={'fixture': 'older run'})
            run = LayoutExtractionRun.objects.create(version=newest, status='RUNNING', total_pages=3,
                total_blocks=7, structure={'fixture': 'large graph not needed by card'})
            cls.expected[contract.pk] = (newest.pk, run.pk)
        other = Company.objects.create(company_name='Other fixture tenant', tax_code='other-fixture')
        Contract.objects.create(company=other, contract_code='DASH-OTHER-FIXTURE', title='Private fixture')
        Contract.objects.create(company=cls.company, contract_code='DASH-EMPTY-FIXTURE', title='No version fixture')

    def test_latest_cards_use_three_queries_without_graph_payloads(self):
        with self.assertNumQueries(3):
            contracts = dashboard_contracts(self.user)
            self.assertEqual(len(contracts), 4)
            for contract in contracts:
                version = contract.latest_version
                run = latest_run(version)
                if contract.pk in self.expected:
                    self.assertEqual((version.pk, run.pk), self.expected[contract.pk])
                    self.assertEqual(run.status, 'RUNNING')
                    self.assertEqual((run.total_pages, run.total_blocks), (3, 7))
                    self.assertIn('structure', run.get_deferred_fields())
                    self.assertEqual(len(list(contract.versions.all())), 1)
                    self.assertEqual(len(list(version.layout_extraction_runs.all())), 1)
                else:
                    self.assertIsNone(version)
                    self.assertIsNone(run)

    def test_tenant_scope_is_preserved(self):
        self.assertFalse(any(c.contract_code == 'DASH-OTHER-FIXTURE' for c in dashboard_contracts(self.user)))
