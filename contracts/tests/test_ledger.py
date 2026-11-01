import base64
import hashlib
import json
import tempfile
import uuid
import os
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch, Mock
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from django.test import TestCase, Client, override_settings
from django.utils import timezone
from contracts.models import Company, Role, User, Contract, ContractVersion, ContractFile, ContractSigningCredential, ContractLedgerOperation
from contracts.services.document_encryption_service import encrypt_pdf
from contracts.services import ledger_service as ledger


class LedgerTests(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.private = rsa.generate_private_key(public_exponent=65537, key_size=3072)
        cls.pem = cls.private.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.override = override_settings(MEDIA_ROOT=self.temp.name, MEDIA_URL='/media/')
        self.override.enable()
        self.addCleanup(self.override.disable)
        self.a = Company.objects.create(company_name='A', tax_code='A')
        self.b = Company.objects.create(company_name='B', tax_code='B')
        self.role = Role.objects.create(role_name='SIGNER')
        manager_role = Role.objects.create(role_name='MANAGER')
        self.user = User.objects.create_user(username='signer', company=self.a, role=self.role)
        self.reviewer = User.objects.create_user(username='reviewer', company=self.a, role=manager_role)
        self.other = User.objects.create_user(username='foreign', company=self.b, role=self.role)
        self.contract = Contract.objects.create(company=self.a, contract_code='A1', title='A contract')
        self.version = ContractVersion.objects.create(contract=self.contract, version_number=1)
        self.path = Path(self.temp.name) / 'source.bin'
        self.raw = b'%PDF-1.7\x00\xff\x80 original bytes'
        self.path.write_bytes(encrypt_pdf(self.raw))
        ContractFile.objects.create(version=self.version, file_path='/media/source.bin', file_name='source.pdf')
        _, pem, fingerprint = ledger.public_key(self.pem)
        self.key = ContractSigningCredential.objects.create(owner=self.user, company=self.a, public_key_pem=pem,
            fingerprint=fingerprint, status='ACTIVE', approved_by=self.reviewer, approved_at=timezone.now())
        self.client.force_login(self.user)

    def post(self, url, data):
        return self.client.post(url, json.dumps(data), content_type='application/json')

    def prepare(self, version=None):
        return ledger.prepare(self.user, version or self.version, 'SIGN', self.key.pk)

    def signature(self, op):
        return base64.b64encode(self.private.sign(op.manifest.encode(), padding.PKCS1v15(), hashes.SHA256())).decode()

    def accepted(self):
        op = self.prepare()
        return ledger.accept(self.user, op.pk, self.signature(op))

    def receipt(self, op):
        return {'payload': op.ledger_payload, 'payload_hash': hashlib.sha256(op.ledger_payload.encode()).hexdigest(),
                'company_id': op.company_id, 'tx_id': 'a' * 64, 'block_number': 4, 'block_hash': 'b' * 64}

    def test_anonymous_api_denied(self):
        self.client.logout()
        self.assertEqual(self.client.get('/api/ledger/operations/').status_code, 401)

    def test_company_scope_on_read_and_prepare(self):
        self.client.force_login(self.other)
        self.assertEqual(self.client.get('/api/ledger/operations/', {'version_id': self.version.pk}).status_code, 404)
        self.assertEqual(self.post('/api/ledger/operations/', {'version_id': self.version.pk, 'credential_id': self.key.pk}).status_code, 404)

    def test_companyless_user_cannot_prepare(self):
        self.user.company = None
        self.user.save()
        with self.assertRaises(Exception): self.prepare()

    def test_owner_cannot_approve_own_key(self):
        self.key.status = 'PENDING'
        self.key.save()
        self.assertEqual(self.post(f'/api/ledger/credentials/{self.key.pk}/', {'action':'approve'}).status_code, 403)

    def test_foreign_manager_cannot_approve_key(self):
        self.reviewer.company = self.b
        self.reviewer.save()
        self.client.force_login(self.reviewer)
        self.assertEqual(self.post(f'/api/ledger/credentials/{self.key.pk}/', {'action':'approve'}).status_code, 404)

    def test_independent_manager_approval(self):
        self.key.status = 'PENDING'
        self.key.save()
        self.client.force_login(self.reviewer)
        self.assertEqual(self.post(f'/api/ledger/credentials/{self.key.pk}/', {'action':'approve'}).status_code, 200)
        self.key.refresh_from_db()
        self.assertEqual(self.key.approved_by_id, self.reviewer.pk)

    def test_wrong_signature_rejected(self):
        op = self.prepare()
        r = self.post(f'/api/ledger/operations/{op.pk}/', {'signature':base64.b64encode(b'fake').decode()})
        self.assertEqual(r.status_code, 400)
        op.refresh_from_db()
        self.assertIsNone(op.accepted_at)

    def test_signature_cannot_move_to_another_version(self):
        op = self.prepare()
        v2 = ContractVersion.objects.create(contract=self.contract, version_number=2)
        ContractFile.objects.create(version=v2, file_path='/media/source.bin')
        op2 = self.prepare(v2)
        self.assertEqual(self.post(f'/api/ledger/operations/{op2.pk}/', {'signature':self.signature(op)}).status_code, 400)

    def test_modified_file_rejected_before_queue(self):
        op = self.prepare()
        self.path.write_bytes(encrypt_pdf(b'modified'))
        self.assertEqual(self.post(f'/api/ledger/operations/{op.pk}/', {'signature':self.signature(op)}).status_code, 400)

    def test_duplicate_prepare_and_submit_are_idempotent(self):
        op = self.prepare()
        self.assertEqual(self.prepare().pk, op.pk)
        sig = self.signature(op)
        ledger.accept(self.user, op.pk, sig)
        ledger.accept(self.user, op.pk, sig)
        self.assertEqual(ContractLedgerOperation.objects.count(), 1)

    def test_expired_challenge_rejected(self):
        op = self.prepare()
        op.expires_at = timezone.now() - timedelta(seconds=1)
        op.save()
        self.assertEqual(self.post(f'/api/ledger/operations/{op.pk}/', {'signature':self.signature(op)}).status_code, 400)

    def test_revoked_key_blocks_submission(self):
        op = self.prepare()
        self.key.status = 'REVOKED'
        self.key.save()
        self.assertEqual(self.post(f'/api/ledger/operations/{op.pk}/', {'signature':self.signature(op)}).status_code, 404)

    def test_failed_gateway_keeps_durable_request_without_fake_tx(self):
        op = self.accepted()
        claimed = ledger.claim()
        missing = requests.HTTPError(response=Mock(status_code=404))
        with patch('contracts.services.ledger_service.gateway', side_effect=[missing, requests.ConnectionError('offline')]):
            ledger.process(claimed)
        op.refresh_from_db()
        self.assertEqual(op.state, 'FAILED')
        self.assertIsNone(op.tx_id)
        self.assertEqual(op.attempts, 1)
        self.assertIsNone(ledger.claim())

    def test_commit_recovery_does_not_submit_twice(self):
        op = self.accepted()
        claimed = ledger.claim()
        with patch('contracts.services.ledger_service.gateway', return_value=self.receipt(op)) as gateway:
            ledger.process(claimed)
        self.assertEqual(gateway.call_count, 1)
        op.refresh_from_db()
        self.assertEqual(op.state, 'CONFIRMED')
        self.assertEqual(op.tx_id, 'a'*64)

    def test_stale_worker_cannot_confirm_after_lease_changes(self):
        op = self.accepted()
        claimed = ledger.claim()
        ContractLedgerOperation.objects.filter(pk=op.pk).update(lease_token=uuid.uuid4())
        with patch('contracts.services.ledger_service.gateway', return_value=self.receipt(op)): ledger.process(claimed)
        op.refresh_from_db()
        self.assertNotEqual(op.state, 'CONFIRMED')

    def test_conflicting_ledger_payload_never_confirms(self):
        op = self.accepted()
        claimed = ledger.claim()
        with patch('contracts.services.ledger_service.gateway', return_value={**self.receipt(op), 'payload':'different'}): ledger.process(claimed)
        op.refresh_from_db()
        self.assertEqual(op.state, 'FAILED')

    def test_verify_separates_upload_signature_and_ledger(self):
        from django.core.files.uploadedfile import SimpleUploadedFile
        op = self.accepted()
        with patch('contracts.services.ledger_service.gateway', return_value=self.receipt(op)):
            r = self.client.post('/api/ledger/verify/', {'version_id':self.version.pk, 'file':SimpleUploadedFile('modified.pdf', b'tampered')})
        self.assertEqual(r.status_code, 200)
        result = r.json()['results'][0]
        self.assertTrue(result['signature_valid'])
        self.assertTrue(result['file_integrity'])
        self.assertTrue(result['ledger_verified'])
        self.assertFalse(result['uploaded_file_matches'])

    def test_ledger_outage_is_unknown_not_verified(self):
        self.accepted()
        with patch('contracts.services.ledger_service.gateway', side_effect=requests.ConnectionError()):
            r = self.post('/api/ledger/verify/', {'version_id':self.version.pk})
        self.assertIsNone(r.json()['results'][0]['ledger_verified'])

    def test_missing_source_does_not_hide_valid_uploaded_file_signature(self):
        from django.core.files.uploadedfile import SimpleUploadedFile
        op = self.accepted()
        self.path.unlink()
        with patch('contracts.services.ledger_service.gateway', return_value=self.receipt(op)):
            r = self.client.post('/api/ledger/verify/', {'version_id':self.version.pk, 'file':SimpleUploadedFile('source.pdf', self.raw)})
        self.assertEqual(r.status_code, 200)
        result = r.json()['results'][0]
        self.assertIsNone(result['file_integrity'])
        self.assertTrue(result['uploaded_file_matches'])
        self.assertTrue(result['signature_valid'])
        self.assertTrue(result['ledger_verified'])

    def test_wrong_company_receipt_cannot_confirm(self):
        op = self.accepted()
        claimed = ledger.claim()
        with patch('contracts.services.ledger_service.gateway', return_value={**self.receipt(op), 'company_id':self.b.pk}): ledger.process(claimed)
        op.refresh_from_db()
        self.assertEqual(op.state, 'FAILED')

    def test_binary_file_bytes_hashed_without_text_conversion(self):
        self.assertEqual(ledger.snapshot(self.version)['document_hash'], hashlib.sha256(self.raw).hexdigest())

    def test_accepted_version_cannot_be_renumbered(self):
        from django.core.exceptions import ValidationError
        self.accepted()
        self.version.version_number = 99
        with self.assertRaises(ValidationError): self.version.save()

    def test_accepted_contract_cannot_move_to_another_company(self):
        from django.core.exceptions import ValidationError
        self.accepted()
        self.contract.company = self.b
        with self.assertRaises(ValidationError): self.contract.save()

    def test_accepted_version_cannot_replace_or_add_files(self):
        from django.core.exceptions import ValidationError
        self.accepted()
        file = self.version.files.first()
        file.file_path = '/media/other.bin'
        with self.assertRaises(ValidationError): file.save()
        with self.assertRaises(ValidationError): ContractFile.objects.create(version=self.version, file_path='/media/new.bin')

    def test_cookie_requests_require_csrf(self):
        client = Client(enforce_csrf_checks=True)
        client.force_login(self.user)
        self.assertEqual(client.post('/api/ledger/operations/', '{}', content_type='application/json').status_code, 403)

    def test_workspace_renders(self):
        self.assertEqual(self.client.get('/contract-ledger/').status_code, 200)

    def test_small_rsa_key_is_rejected(self):
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        pem = key.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode()
        self.assertEqual(self.post('/api/ledger/credentials/', {'public_key_pem':pem}).status_code, 400)

    @unittest.skipUnless(os.getenv('LEDGER_LIVE_SMOKE') == '1', 'Opt-in isolated database + live Fabric smoke test')
    def test_live_commit_and_lost_response_recovery(self):
        op = self.accepted()
        claimed = ledger.claim()
        original = ledger.gateway
        commits = []
        def lose_response(path, payload=None):
            result = original(path, payload)
            if payload is not None:
                commits.append(result)
                raise requests.ConnectionError('Simulated response loss after real commit')
            return result
        with override_settings(FABRIC_GATEWAY_TOKEN=os.getenv('FABRIC_GATEWAY_TOKEN')):
            with patch('contracts.services.ledger_service.gateway', side_effect=lose_response): ledger.process(claimed)
            op.refresh_from_db()
            self.assertEqual(op.state, 'FAILED')
            self.assertEqual(len(commits), 1)
            ContractLedgerOperation.objects.filter(pk=op.pk).update(next_attempt_at=timezone.now())
            second = ledger.claim()
            with patch('contracts.services.ledger_service.gateway', wraps=original) as calls: ledger.process(second)
            self.assertEqual(calls.call_count, 1)
            op.refresh_from_db()
            self.assertEqual(op.state, 'CONFIRMED')
            self.assertEqual(op.tx_id, commits[0]['tx_id'])
            receipt = original('/enterprise/operations/' + str(op.pk))
            self.assertEqual(receipt['payload'], op.ledger_payload)
            self.assertTrue(ledger.verify_signature(self.pem, op.signature_base64, op.manifest))
            print('Live RSA commit + recovery PASS:', op.tx_id, 'block', op.block_number)
