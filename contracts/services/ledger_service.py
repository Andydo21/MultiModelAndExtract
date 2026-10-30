from contracts.repositories import ledger_workflow_repository
from contracts.repositories import ledger_data_repository as data_repository
'Version-bound signing and a durable, idempotent Fabric outbox.'
import base64
import hashlib
import json
import re
import uuid
from datetime import timedelta
from pathlib import Path
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.exceptions import InvalidSignature
from cryptography.exceptions import InvalidTag
from django.conf import settings
from django.core.exceptions import PermissionDenied
from django.db import transaction
from django.db.models import Q
from django.utils import timezone
from contracts.services.document_encryption_service import decrypt_pdf

def canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'))

def audit(user, action, resource):
    data_repository.create_audit_log_for_audit(action, canonical, resource, user)

def manager(user):
    return user.is_superuser or bool(user.role and user.role.role_name.upper() in {'ADMIN', 'MANAGER'})

def allowed(user, permission):
    return manager(user) or bool(user.role and (user.role.role_name.upper() == 'SIGNER' or user.role.permissions.filter(permission_code=permission).exists()))

def public_key(pem):
    key = serialization.load_pem_public_key(pem.encode())
    if not isinstance(key, rsa.RSAPublicKey) or key.key_size < 3072 or key.public_numbers().e != 65537:
        raise ValueError('Cần public key RSA ít nhất 3072 bit, exponent 65537.')
    normalized = key.public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode()
    der = key.public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    return (key, normalized, hashlib.sha256(der).hexdigest())

def verify_signature(pem, signature, manifest):
    try:
        key, _, _ = public_key(pem)
        key.verify(base64.b64decode(signature, validate=True), manifest.encode('utf-8'), padding.PKCS1v15(), hashes.SHA256())
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False

def snapshot(version):
    root = Path(settings.MEDIA_ROOT).resolve()
    result = []
    for file in data_repository.order_by_files_for_snapshot(version):
        relative = file.file_path
        if relative.startswith(settings.MEDIA_URL):
            relative = relative[len(settings.MEDIA_URL):]
        path = (root / relative.lstrip('/')).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Đường dẫn file nằm ngoài kho hợp đồng.')
        try:
            data = decrypt_pdf(path.read_bytes())
        except (OSError, InvalidTag, ValueError) as exc:
            raise ValueError('Không đọc/giải mã được file nguồn của phiên bản.') from exc
        result.append({'file_id': file.pk, 'sha256': hashlib.sha256(data).hexdigest(), 'size': len(data)})
    if not result:
        raise ValueError('Phiên bản chưa có file để kiểm chứng.')
    digest = result[0]['sha256'] if len(result) == 1 else hashlib.sha256(canonical(result).encode()).hexdigest()
    return {'files': result, 'document_hash': digest}

def credential_for(user, key_id):
    key = data_repository.get_contract_signing_credential_for_credential_for(key_id, user)
    if key.owner_id == key.approved_by_id or not key.approved_by_id:
        raise PermissionDenied('Public key phải được người quản lý khác duyệt.')
    return key

def workflow_step(user, version, step_id):
    if not step_id:
        return
    row, blocked = ledger_workflow_repository.signing_step(step_id)
    if not row or row[2] != version.pk:
        raise PermissionDenied('Bước ký không thuộc phiên bản này.')
    if row[1] == 'REJECTED' or (not user.is_superuser and row[0] and (row[0] != user.role_id)):
        raise PermissionDenied('Không được ký bước workflow này.')
    if blocked:
        raise PermissionDenied('Bước tiên quyết chưa được duyệt.')

def prepare(user, version, kind, credential_id=None, step_id=0):
    if not user.company_id or user.company_id != version.contract.company_id:
        raise PermissionDenied('Cần tài khoản thuộc công ty của hợp đồng.')
    if not allowed(user, 'ledger.sign' if kind == 'SIGN' else 'ledger.anchor'):
        raise PermissionDenied('Không có quyền ký/ghi ledger.')
    key = credential_for(user, credential_id) if kind == 'SIGN' else None
    workflow_step(user, version, step_id)
    source = snapshot(version)
    semantic = canonical({'version': version.pk, 'company': user.company_id, 'kind': kind, 'actor': user.pk, 'key': key.pk if key else None, 'step': step_id})
    idem = hashlib.sha256(semantic.encode()).hexdigest()
    with transaction.atomic():
        data_repository.get_record_for_prepare(version)
        existing = data_repository.first_contract_ledger_operation_for_prepare(idem)
        if existing:
            manifest = json.loads(existing.manifest)
            if manifest['document_hash'] != source['document_hash']:
                raise ValueError('Nội dung phiên bản đã đổi sau khi chuẩn bị ký. Hãy tạo phiên bản mới.')
            if not existing.accepted_at and existing.expires_at < timezone.now():
                existing.manifest = canonical({**manifest, 'issued_at': timezone.now().isoformat(), 'expires_at': (timezone.now() + timedelta(minutes=10)).isoformat()})
                existing.expires_at = timezone.now() + timedelta(minutes=10)
                data_repository.save_existing_for_prepare(existing)
            return existing
        now = timezone.now()
        op_id = uuid.uuid4()
        manifest = {'schema': 'contract-ledger-v1', 'operation_id': str(op_id), 'kind': kind, 'company_id': user.company_id, 'contract_id': version.contract_id, 'version_id': version.pk, 'version_number': version.version_number, 'signer_id': user.pk, 'step_id': step_id, 'credential_fingerprint': key.fingerprint if key else '', 'issued_at': now.isoformat(), 'expires_at': (now + timedelta(minutes=10)).isoformat(), **source}
        op = data_repository.create_contract_ledger_operation_for_prepare(canonical, idem, key, kind, manifest, now, op_id, step_id, timedelta, user, version)
        audit(user, 'LEDGER_PREPARE', str(op.id))
        return op

def accept(user, op_id, signature=''):
    with transaction.atomic():
        op = data_repository.get_contract_ledger_operation_for_accept(op_id, user)
        if op.accepted_at:
            if op.kind == 'SIGN' and signature != op.signature_base64:
                raise ValueError('Yêu cầu đã được ký bằng một chữ ký khác.')
            return op
        if op.expires_at < timezone.now():
            raise ValueError('Yêu cầu ký đã hết hạn. Chuẩn bị lại để ký.')
        if not allowed(user, 'ledger.sign' if op.kind == 'SIGN' else 'ledger.anchor'):
            raise PermissionDenied('Quyền thao tác đã bị thu hồi.')
        if user.company_id != op.version.contract.company_id:
            raise PermissionDenied('Công ty của hợp đồng đã thay đổi.')
        manifest = json.loads(op.manifest)
        if (manifest['contract_id'], manifest['version_number']) != (op.version.contract_id, op.version.version_number):
            raise ValueError('Thông tin phiên bản đã thay đổi. Cần chuẩn bị lại trên phiên bản mới.')
        if snapshot(op.version)['document_hash'] != manifest['document_hash']:
            raise ValueError('File đã thay đổi. Cần tạo phiên bản mới.')
        workflow_step(user, op.version, op.step_id)
        if op.kind == 'SIGN':
            credential_for(user, op.credential_id)
            if not verify_signature(op.credential.public_key_pem, signature, op.manifest):
                raise ValueError('Chữ ký không hợp lệ hoặc không thuộc phiên bản này.')
        op.accepted_at = timezone.now()
        op.signature_base64 = signature if op.kind == 'SIGN' else ''
        op.ledger_payload = canonical({'manifest': op.manifest, 'signature': op.signature_base64, 'public_key_pem': op.credential.public_key_pem if op.credential_id else '', 'credential_owner': user.pk, 'credential_company': user.company_id, 'credential_approver': op.credential.approved_by_id if op.credential_id else None})
        data_repository.save_op_for_accept(op)
        audit(user, 'LEDGER_QUEUED', str(op.id))
        return op

def gateway(path, payload=None):
    base = getattr(settings, 'FABRIC_GATEWAY_URL', 'http://fabric-gateway:5000').rstrip('/')
    headers = {'X-Contract-Gateway-Token': getattr(settings, 'FABRIC_GATEWAY_TOKEN', '')}
    if not headers['X-Contract-Gateway-Token']:
        raise RuntimeError('Thiếu FABRIC_GATEWAY_TOKEN.')
    r = requests.post(base + path, json=payload, headers=headers, timeout=80) if payload is not None else requests.get(base + path, headers=headers, timeout=15)
    r.raise_for_status()
    return r.json()

def claim():
    now = timezone.now()
    with transaction.atomic():
        rows = data_repository.order_by_contract_ledger_operation_for_claim(Q, now)
        op = rows.first()
        if not op:
            return None
        op.state = 'SUBMITTED'
        op.attempts += 1
        op.lease_until = now + timedelta(minutes=3)
        op.lease_token = uuid.uuid4()
        data_repository.save_op_for_accept(op)
        return op

def valid_receipt(op, record):
    return record.get('payload') == op.ledger_payload and record.get('company_id') == op.company_id and (record.get('payload_hash') == hashlib.sha256(op.ledger_payload.encode()).hexdigest()) and bool(re.fullmatch('[0-9a-f]{64}', record.get('tx_id', ''))) and bool(re.fullmatch('[0-9a-f]{64}', record.get('block_hash', ''))) and (type(record.get('block_number')) is int) and (record['block_number'] >= 0)

def process(op):
    try:
        existing = None
        try:
            existing = gateway('/enterprise/operations/' + str(op.id))
        except requests.HTTPError as exc:
            if exc.response is None or exc.response.status_code != 404:
                raise
        if existing is None:
            op.actor.refresh_from_db()
            if not op.actor.is_active or not allowed(op.actor, 'ledger.sign' if op.kind == 'SIGN' else 'ledger.anchor'):
                raise PermissionDenied('Quyền người ký đã bị thu hồi.')
            if op.actor.company_id != op.company_id or op.version.contract.company_id != op.company_id:
                raise PermissionDenied('Công ty không còn khớp yêu cầu.')
            manifest = json.loads(op.manifest)
            if (manifest['contract_id'], manifest['version_number']) != (op.version.contract_id, op.version.version_number):
                raise ValueError('Thông tin phiên bản đã thay đổi; không ghi ledger.')
            if snapshot(op.version)['document_hash'] != json.loads(op.manifest)['document_hash']:
                raise ValueError('File nguồn đã thay đổi; không ghi ledger.')
            if op.kind == 'SIGN':
                credential_for(op.actor, op.credential_id)
            workflow_step(op.actor, op.version, op.step_id)
            existing = gateway('/enterprise/commit', {'payload': op.ledger_payload})
        if not valid_receipt(op, existing):
            raise ValueError('Bằng chứng ledger không khớp yêu cầu.')
        with transaction.atomic():
            current = data_repository.get_contract_ledger_operation_for_process(op)
            if current.lease_token != op.lease_token:
                return
            current.state = 'CONFIRMED'
            current.tx_id = existing['tx_id']
            current.block_number = existing['block_number']
            current.block_hash = existing['block_hash']
            current.confirmed_at = timezone.now()
            current.last_error = ''
            current.lease_until = None
            data_repository.save_current_for_process(current)
            audit(current.actor, 'LEDGER_CONFIRMED', str(current.id))
    except Exception as exc:
        data_repository.update_contract_ledger_operation_for_process(exc, op, timedelta, timezone)

def sync_workflow(op):
    if not op.step_id or op.kind != 'SIGN':
        return
    ledger_workflow_repository.synchronize_confirmed_signature(op)
    data_repository.update_contract_ledger_operation_for_sync_workflow(op)
