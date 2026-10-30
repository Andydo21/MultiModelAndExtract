import uuid
from django.conf import settings
from django.db import models


class ContractSigningCredential(models.Model):
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.PROTECT, related_name='signing_credentials')
    company = models.ForeignKey('Company', on_delete=models.PROTECT)
    public_key_pem = models.TextField()
    fingerprint = models.CharField(max_length=64, unique=True)
    status = models.CharField(max_length=16, default='PENDING', choices=[(x, x) for x in ['PENDING', 'ACTIVE', 'REVOKED']])
    approved_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.PROTECT, null=True, related_name='approved_signing_credentials')
    approved_at = models.DateTimeField(null=True)
    revoked_at = models.DateTimeField(null=True)
    created_at = models.DateTimeField(auto_now_add=True)


class ContractLedgerOperation(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    version = models.ForeignKey('ContractVersion', on_delete=models.PROTECT, related_name='ledger_operations')
    company = models.ForeignKey('Company', on_delete=models.PROTECT)
    actor = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.PROTECT)
    credential = models.ForeignKey(ContractSigningCredential, on_delete=models.PROTECT, null=True)
    kind = models.CharField(max_length=12, choices=[('ANCHOR', 'ANCHOR'), ('SIGN', 'SIGN')])
    step_id = models.PositiveBigIntegerField(default=0)
    idempotency_key = models.CharField(max_length=64, unique=True)
    manifest = models.TextField()
    signature_base64 = models.TextField(blank=True)
    ledger_payload = models.TextField(blank=True)
    state = models.CharField(max_length=16, default='PENDING', db_index=True,
        choices=[(x, x) for x in ['PENDING', 'SUBMITTED', 'CONFIRMED', 'FAILED']])
    accepted_at = models.DateTimeField(null=True)
    expires_at = models.DateTimeField()
    attempts = models.PositiveIntegerField(default=0)
    next_attempt_at = models.DateTimeField(db_index=True)
    lease_until = models.DateTimeField(null=True)
    lease_token = models.UUIDField(null=True)
    last_error = models.TextField(blank=True)
    tx_id = models.CharField(max_length=64, null=True, unique=True)
    block_number = models.PositiveBigIntegerField(null=True)
    block_hash = models.CharField(max_length=64, blank=True)
    confirmed_at = models.DateTimeField(null=True)
    workflow_synced = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
