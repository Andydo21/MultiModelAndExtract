from contracts.repositories import ledger_api_repository
from contracts.repositories import ledger_workflow_repository
from contracts.repositories import ledger_operations_repository as workflow_repository
import hashlib
import json
from django.core.exceptions import PermissionDenied, ObjectDoesNotExist
from django.db import transaction
from django.utils import timezone
from contracts.services import ledger_service as ledger
from contracts.services.results import ServiceResult
from contracts.repositories.ledger_api_repository import version_for

def serialize(op):
    return {'id': str(op.pk), 'kind': op.kind, 'state': op.state, 'version_id': op.version_id, 'actor': op.actor.username, 'manifest': op.manifest, 'signature': op.signature_base64, 'credential_id': op.credential_id, 'attempts': op.attempts, 'last_error': op.last_error, 'tx_id': op.tx_id, 'block_number': op.block_number, 'block_hash': op.block_hash, 'accepted': bool(op.accepted_at), 'workflow_synced': op.workflow_synced, 'created_at': op.created_at.isoformat()}

def workspace(user, query):
    versions, selected = ledger_api_repository.workspace_versions(user, query.get('version_id'))
    return ServiceResult({'versions': versions, 'selected': selected, 'can_manage': ledger.manager(user), 'can_sign': ledger.allowed(user, 'ledger.sign')}, template='contracts/ledger.html')

def credentials(user, method, data):
    user = user
    if method == 'GET':
        keys = ledger_api_repository.credentials_for(user, ledger.manager(user))
        return ServiceResult({'keys': [{'id': k.pk, 'owner': k.owner.username, 'owner_id': k.owner_id, 'fingerprint': k.fingerprint, 'status': k.status, 'public_key_pem': k.public_key_pem, 'approved_by': k.approved_by_id, 'can_approve': ledger.manager(user) and k.owner_id != user.pk and (k.status == 'PENDING')} for k in keys]})
    if not user.company_id or not ledger.allowed(user, 'ledger.sign'):
        raise PermissionDenied('Cần quyền ký và công ty để đăng ký public key.')
    _, normalized, fingerprint = ledger.public_key(data['public_key_pem'])
    with transaction.atomic():
        key, created = workflow_repository.get_or_create_contract_signing_credential_for_credentials(fingerprint, normalized, user)
        if key.owner_id != user.pk or key.company_id != user.company_id:
            raise PermissionDenied('Public key đã thuộc tài khoản khác.')
        if created:
            ledger.audit(user, 'SIGNING_KEY_REGISTERED', str(key.pk))
    return ServiceResult({'id': key.pk, 'status': key.status, 'fingerprint': key.fingerprint}, status=201 if created else 200)

def credential_action(key_id, user, data):
    user = user
    with transaction.atomic():
        key = workflow_repository.get_contract_signing_credential_for_credential_action(key_id)
        if not user.is_superuser and key.company_id != user.company_id:
            raise ObjectDoesNotExist()
        action = data['action']
        if action == 'approve':
            if not ledger.manager(user) or key.owner_id == user.pk:
                raise PermissionDenied('Người quản lý khác phải duyệt khóa.')
            if key.status != 'PENDING':
                raise ValueError('Chỉ duyệt khóa đang chờ.')
            key.status, key.approved_by, key.approved_at = ('ACTIVE', user, timezone.now())
        elif action == 'revoke':
            if key.owner_id != user.pk and (not ledger.manager(user)):
                raise PermissionDenied('Không có quyền thu hồi khóa.')
            key.status, key.revoked_at = ('REVOKED', timezone.now())
        else:
            raise ValueError('Action không hợp lệ.')
        workflow_repository.save_record_for_credential_action(key)
        ledger.audit(user, 'SIGNING_KEY_' + action.upper(), str(key.pk))
    return ServiceResult({'id': key.pk, 'status': key.status})

def operations(user, method, data, query):
    if method == 'GET':
        version = version_for(user, query['version_id']) if query.get('version_id') else None
        return ServiceResult({'operations': [serialize(op) for op in ledger_api_repository.operation_list(user, version)]})
    data = data
    version = version_for(user, data['version_id'])
    kind = data.get('kind', 'SIGN')
    if kind not in {'SIGN', 'ANCHOR'}:
        raise ValueError('Kind không hợp lệ.')
    step_id = int(data.get('step_id') or 0)
    if step_id < 0:
        raise ValueError('Step ID không hợp lệ.')
    if kind == 'ANCHOR' and step_id:
        raise ValueError('Bước workflow cần chữ ký, không chỉ anchor.')
    op = ledger.prepare(user, version, kind, data.get('credential_id'), step_id)
    if kind == 'ANCHOR':
        op = ledger.accept(user, op.pk)
    return ServiceResult(serialize(op), status=200 if op.accepted_at else 201)

def workflow_steps(user, query):
    from django.db import connections
    version = version_for(user, query['version_id'])
    steps = [{'id': row[0], 'name': row[1], 'status': row[2]} for row in ledger_workflow_repository.signature_steps(version) if user.is_superuser or not row[3] or row[3] == user.role_id]
    return ServiceResult({'steps': steps})

def operation_action(operation_id, user, data):
    op = ledger_api_repository.operation_for(user, operation_id)
    data = data
    if data.get('action') == 'retry':
        if op.actor_id != user.pk and (not ledger.manager(user)):
            raise PermissionDenied('Không có quyền retry.')
        if op.state != 'FAILED' or not op.accepted_at:
            raise ValueError('Chỉ retry yêu cầu đã gửi và thất bại.')
        with transaction.atomic():
            op = workflow_repository.get_contract_ledger_operation_for_operation_action(op)
            if op.state != 'FAILED' or (op.lease_until and op.lease_until > timezone.now()):
                raise ValueError('Yêu cầu đang được xử lý.')
            op.state, op.attempts, op.next_attempt_at = ('PENDING', 0, timezone.now())
            workflow_repository.save_record_for_operation_action(op)
            ledger.audit(user, 'LEDGER_RETRY', str(op.id))
    else:
        if op.actor_id != user.pk:
            raise PermissionDenied('Chỉ người chuẩn bị yêu cầu được ký.')
        op = ledger.accept(user, op.pk, data.get('signature', ''))
    return ServiceResult(serialize(op))

def verify(user, data, form, uploads, content_type):
    version_id = form.get('version_id') if content_type.startswith('multipart/') else data['version_id']
    version = version_for(user, version_id)
    rows = ledger_api_repository.operation_list(user, version, accepted_only=True)
    uploaded = uploads.get('file')
    uploaded_hash = None
    if uploaded:
        h = hashlib.sha256()
        for chunk in uploaded.chunks():
            h.update(chunk)
        uploaded_hash = h.hexdigest()
    source_error = None
    try:
        current = ledger.snapshot(version)
    except ValueError as exc:
        current = None
        source_error = str(exc)
    results = []
    for op in rows:
        manifest = json.loads(op.manifest)
        file_id = form.get('file_id')
        expected = [f for f in manifest['files'] if not file_id or str(f['file_id']) == file_id]
        item = {'operation_id': str(op.pk), 'kind': op.kind, 'state': op.state, 'file_integrity': current['document_hash'] == manifest['document_hash'] if current else None, 'uploaded_file_matches': any((f['sha256'] == uploaded_hash for f in expected)) if uploaded else None, 'signature_valid': ledger.verify_signature(op.credential.public_key_pem, op.signature_base64, op.manifest) if op.kind == 'SIGN' else None, 'credential_currently_active': op.credential.status == 'ACTIVE' if op.credential_id else None, 'ledger_verified': None, 'tx_id': op.tx_id, 'block_number': op.block_number}
        try:
            record = ledger.gateway('/enterprise/operations/' + str(op.pk))
            item['ledger_verified'] = ledger.valid_receipt(op, record) and (not op.tx_id or record.get('tx_id') == op.tx_id)
            item['tx_id'], item['block_number'] = (record.get('tx_id'), record.get('block_number'))
        except Exception:
            item['ledger_error'] = 'Chưa đọc được bằng chứng ledger; không xác nhận blockchain.'
        results.append(item)
    ledger.audit(user, 'LEDGER_VERIFIED', str(version.pk))
    return ServiceResult({'version_id': version.pk, 'uploaded_hash': uploaded_hash, 'source_error': source_error, 'results': results, 'trust_model': 'application_approved_RSA_key_not_qualified_CA_signature'})

