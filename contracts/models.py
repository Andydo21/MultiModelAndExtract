from django.db import models
from django.contrib.auth.models import AbstractUser
from django.conf import settings
import uuid
from .ledger_models import ContractSigningCredential, ContractLedgerOperation
from django.core.exceptions import ValidationError


class ContractKnowledgeCase(models.Model):
    """A2 expert knowledge, bound to a source version and its company."""
    case_id = models.UUIDField(default=uuid.uuid4, editable=False, db_index=True)
    revision = models.PositiveIntegerField(default=1)
    version = models.ForeignKey('ContractVersion', on_delete=models.PROTECT, related_name='knowledge_cases')
    title = models.CharField(max_length=255)
    clause_type = models.CharField(max_length=100, blank=True)
    issue = models.TextField()
    reasoning = models.TextField(blank=True)
    resolution = models.TextField(blank=True)
    verification = models.TextField(blank=True)
    applicability = models.TextField(blank=True)
    source_reference = models.CharField(max_length=255, blank=True)
    evidence_snapshot = models.JSONField(default=dict, blank=True)
    author = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.PROTECT, related_name='contract_knowledge_authored')
    reviewer = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.PROTECT, null=True, blank=True,
                                 related_name='contract_knowledge_reviewed')
    status = models.CharField(max_length=20, default='draft', db_index=True,
                              choices=[('draft', 'Nháp'), ('approved', 'Đã duyệt'), ('deprecated', 'Ngừng sử dụng')])
    approved_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']
        constraints = [models.UniqueConstraint(fields=['case_id', 'revision'], name='unique_contract_case_revision')]


class KnowledgeSearchLog(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.PROTECT)
    version = models.ForeignKey('ContractVersion', null=True, on_delete=models.PROTECT)
    query = models.TextField()
    scope = models.CharField(max_length=20, default='version')
    generated = models.BooleanField(default=False)
    status = models.CharField(max_length=20, default='RUNNING')
    result = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)


class ContractKnowledgeInquiry(models.Model):
    version = models.ForeignKey('ContractVersion', on_delete=models.PROTECT)
    requester = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.PROTECT, related_name='knowledge_inquiries')
    assignee = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.PROTECT, related_name='assigned_knowledge_inquiries')
    search = models.ForeignKey(KnowledgeSearchLog, on_delete=models.PROTECT)
    case = models.ForeignKey(ContractKnowledgeCase, null=True, blank=True, on_delete=models.PROTECT, related_name='inquiries')
    question = models.TextField()
    status = models.CharField(max_length=20, default='OPEN', choices=[(s, s) for s in ('OPEN', 'ASSIGNED', 'ANSWERED', 'RESOLVED', 'CANCELLED')])
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [models.UniqueConstraint(fields=['search'], name='one_inquiry_per_search')]

class Company(models.Model):
    company_name = models.CharField(max_length=255, verbose_name="Company Name")
    tax_code = models.CharField(max_length=50, verbose_name="Tax Code")
    status = models.CharField(max_length=50, default='ACTIVE', verbose_name="Status")
    tx_hash = models.CharField(max_length=255, blank=True, null=True, verbose_name="Blockchain Tx Hash")
    block_number = models.BigIntegerField(blank=True, null=True, verbose_name="Blockchain Block Number")
    block_hash = models.CharField(max_length=255, blank=True, null=True, verbose_name="Blockchain Block Hash")

    class Meta:
        verbose_name = 'Company'
        verbose_name_plural = 'Companies'

    def __str__(self):
        return self.company_name


class Permission(models.Model):
    permission_code = models.CharField(max_length=100, unique=True, verbose_name="Permission Code")
    description = models.TextField(blank=True, null=True, verbose_name="Description")

    class Meta:
        verbose_name = 'Permission'
        verbose_name_plural = 'Permissions'

    def __str__(self):
        return self.permission_code


class Role(models.Model):
    role_name = models.CharField(max_length=100, unique=True, verbose_name="Role Name")
    description = models.TextField(blank=True, null=True, verbose_name="Description")
    permissions = models.ManyToManyField(Permission, related_name='roles', blank=True, verbose_name="Permissions")

    class Meta:
        verbose_name = 'Role'
        verbose_name_plural = 'Roles'

    def __str__(self):
        return self.role_name


class User(AbstractUser):
    company = models.ForeignKey(Company, on_delete=models.SET_NULL, null=True, blank=True, related_name='users', verbose_name="Company")
    role = models.ForeignKey(Role, on_delete=models.SET_NULL, null=True, blank=True, related_name='users', verbose_name="Role")
    password_hash = models.CharField(max_length=255, blank=True, null=True, verbose_name="Password Hash")
    status = models.CharField(max_length=50, default='ACTIVE', verbose_name="Status")
    tx_hash = models.CharField(max_length=255, blank=True, null=True, verbose_name="Blockchain Tx Hash")
    block_number = models.BigIntegerField(blank=True, null=True, verbose_name="Blockchain Block Number")
    block_hash = models.CharField(max_length=255, blank=True, null=True, verbose_name="Blockchain Block Hash")

    class Meta:
        verbose_name = 'User'
        verbose_name_plural = 'Users'


class Tag(models.Model):
    tag_name = models.CharField(max_length=100, unique=True, verbose_name="Tag Name")

    class Meta:
        verbose_name = 'Tag'
        verbose_name_plural = 'Tags'

    def __str__(self):
        return self.tag_name


class Contract(models.Model):
    STATUS_CHOICES = (
        ('DRAFT', 'Draft'),
        ('ANALYZING', 'Analyzing'),
        ('ANALYZED', 'Analyzed'),
        ('APPROVED', 'Approved'),
    )
    company = models.ForeignKey(Company, on_delete=models.SET_NULL, null=True, blank=True, related_name='contracts', verbose_name="Company")
    contract_code = models.CharField(max_length=50, unique=True, verbose_name="Contract Code")
    title = models.CharField(max_length=255, verbose_name="Title")
    contract_type = models.CharField(max_length=100, blank=True, null=True, verbose_name="Contract Type")
    start_date = models.DateField(blank=True, null=True, verbose_name="Start Date")
    end_date = models.DateField(blank=True, null=True, verbose_name="End Date")
    contract_value = models.DecimalField(max_digits=18, decimal_places=2, blank=True, null=True, verbose_name="Contract Value")
    status = models.CharField(max_length=50, choices=STATUS_CHOICES, default='DRAFT', verbose_name="Status")
    tags = models.ManyToManyField(Tag, related_name='contracts', blank=True, verbose_name="Tags")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Created At")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="Updated At")

    class Meta:
        verbose_name = 'Contract'
        verbose_name_plural = 'Contracts'

    def __str__(self):
        return f"{self.contract_code} - {self.title}"

    def save(self, *args, **kwargs):
        if self.pk:
            old = type(self).objects.filter(pk=self.pk).values('company_id').first()
            if old and old['company_id'] != self.company_id and ContractLedgerOperation.objects.filter(version__contract_id=self.pk, accepted_at__isnull=False).exists():
                raise ValidationError('Không đổi công ty của hợp đồng đã có yêu cầu ledger. Cần quy trình chuyển giao riêng.')
        return super().save(*args, **kwargs)

    @property
    def latest_version(self):
        if hasattr(self, '_prefetched_objects_cache') and 'versions' in self._prefetched_objects_cache:
            versions_list = list(self.versions.all())
            if versions_list:
                return sorted(versions_list, key=lambda v: v.version_number, reverse=True)[0]
            return None
        return self.versions.order_by('-version_number').first()

    @property
    def files(self):
        latest = self.latest_version
        if latest:
            if hasattr(latest, '_prefetched_objects_cache') and 'files' in latest._prefetched_objects_cache:
                return latest.files
            return ContractFile.objects.filter(version=latest)
        return ContractFile.objects.none()

    @property
    def clauses(self):
        latest = self.latest_version
        if latest:
            if hasattr(latest, '_prefetched_objects_cache') and 'clauses' in latest._prefetched_objects_cache:
                return latest.clauses
            from ai_extract.models import Clause
            return Clause.objects.filter(version=latest)
        from ai_extract.models import Clause
        return Clause.objects.none()

    @property
    def ai_analyses(self):
        latest = self.latest_version
        if latest:
            if hasattr(latest, '_prefetched_objects_cache') and 'ai_analyses' in latest._prefetched_objects_cache:
                return latest.ai_analyses
            return AIAnalysis.objects.filter(version=latest)
        return AIAnalysis.objects.none()


class ContractVersion(models.Model):
    contract = models.ForeignKey(Contract, on_delete=models.CASCADE, related_name='versions', verbose_name="Contract")
    version_number = models.IntegerField(default=1, verbose_name="Version Number")
    file_hash = models.CharField(max_length=64, blank=True, null=True, verbose_name="File Hash")
    change_summary = models.TextField(blank=True, null=True, verbose_name="Change Summary")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Created At")

    class Meta:
        verbose_name = 'Contract Version'
        verbose_name_plural = 'Contract Versions'

    def __str__(self):
        return f"{self.contract.contract_code} - v{self.version_number}"

    def save(self, *args, **kwargs):
        if self.pk:
            old = type(self).objects.filter(pk=self.pk).values('contract_id', 'version_number').first()
            if old and (old['contract_id'], old['version_number']) != (self.contract_id, self.version_number) and self.ledger_operations.filter(accepted_at__isnull=False).exists():
                raise ValidationError('Phiên bản đã gửi ledger không được đổi hợp đồng hoặc số phiên bản.')
        return super().save(*args, **kwargs)


class ContractFile(models.Model):
    version = models.ForeignKey(ContractVersion, on_delete=models.CASCADE, related_name='files', verbose_name="Contract Version")
    file_name = models.CharField(max_length=255, blank=True, null=True, verbose_name="File Name")
    file_path = models.CharField(max_length=512, verbose_name="File Path")
    file_size = models.BigIntegerField(blank=True, null=True, verbose_name="File Size")
    mime_type = models.CharField(max_length=100, blank=True, null=True, verbose_name="MIME Type")
    uploaded_at = models.DateTimeField(auto_now_add=True, verbose_name="Uploaded At")

    def save(self, *args, **kwargs):
        old = type(self).objects.filter(pk=self.pk).values('version_id', 'file_path').first() if self.pk else None
        affected = {self.version_id, old['version_id'] if old else self.version_id}
        if (not old or (old['version_id'], old['file_path']) != (self.version_id, self.file_path)) and ContractLedgerOperation.objects.filter(version_id__in=affected, accepted_at__isnull=False).exists():
            raise ValidationError('Không thay file của phiên bản đã gửi ledger. Hãy tạo phiên bản mới.')
        return super().save(*args, **kwargs)

    class Meta:
        verbose_name = 'Contract File'
        verbose_name_plural = 'Contract Files'

    def __str__(self):
        return f"File for {self.version.contract.contract_code} v{self.version.version_number} ({self.uploaded_at})"

    @property
    def contract(self):
        return self.version.contract


class ContractContext(models.Model):
    version = models.ForeignKey(ContractVersion, on_delete=models.CASCADE, related_name='contexts', verbose_name="Contract Version")
    context_type = models.CharField(max_length=100, verbose_name="Context Type")
    source = models.CharField(max_length=255, verbose_name="Source")
    content = models.TextField(verbose_name="Content")
    relevance_score = models.DecimalField(max_digits=5, decimal_places=2, default=0.0, verbose_name="Relevance Score")

    class Meta:
        verbose_name = 'Contract Context'
        verbose_name_plural = 'Contract Contexts'

    def __str__(self):
        return f"Context {self.context_type} (v{self.version.version_number})"


class ContextEmbedding(models.Model):
    context = models.OneToOneField(ContractContext, on_delete=models.CASCADE, related_name='embedding', verbose_name="Contract Context")
    vector_id = models.CharField(max_length=255, verbose_name="Vector ID")
    embedding_model = models.CharField(max_length=100, verbose_name="Embedding Model")

    class Meta:
        verbose_name = 'Context Embedding'
        verbose_name_plural = 'Context Embeddings'

    def __str__(self):
        return f"Embedding for context {self.context_id}"





class RiskRule(models.Model):
    SEVERITY_CHOICES = (
        ('HIGH', 'High'),
        ('MEDIUM', 'Medium'),
        ('LOW', 'Low'),
    )
    rule_code = models.CharField(max_length=50, unique=True, verbose_name="Rule Code")
    rule_name = models.CharField(max_length=255, unique=True, verbose_name="Rule Name")
    description = models.TextField(blank=True, null=True, verbose_name="Description")
    severity = models.CharField(max_length=20, choices=SEVERITY_CHOICES, default='MEDIUM', verbose_name="Severity")

    class Meta:
        verbose_name = 'Risk Rule'
        verbose_name_plural = 'Risk Rules'

    def __str__(self):
        return f"{self.rule_name} ({self.severity})"

# Alias for backward compatibility in codebase
Risk = RiskRule


class AIAnalysis(models.Model):
    version = models.ForeignKey(ContractVersion, on_delete=models.CASCADE, related_name='ai_analyses', verbose_name="Contract Version")
    model_name = models.CharField(max_length=100, verbose_name="Model Name")
    overall_score = models.DecimalField(max_digits=5, decimal_places=2, verbose_name="Overall Score")
    risk_level = models.CharField(max_length=20, default='MEDIUM', verbose_name="Risk Level")
    summary = models.TextField(blank=True, null=True, verbose_name="Summary")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Analyzed At")

    class Meta:
        verbose_name = 'AI Analysis'
        verbose_name_plural = 'AI Analyses'

    def __str__(self):
        return f"Analysis ({self.model_name}) - v{self.version.version_number} ({self.created_at.strftime('%Y-%m-%d %H:%M')})"

    @property
    def contract(self):
        return self.version.contract


class RiskFinding(models.Model):
    RISK_LEVEL_CHOICES = (
        ('HIGH', 'High'),
        ('MEDIUM', 'Medium'),
        ('LOW', 'Low'),
    )
    analysis = models.ForeignKey(AIAnalysis, on_delete=models.CASCADE, related_name='findings', verbose_name="AI Analysis")
    clause = models.ForeignKey("ai_extract.Clause", on_delete=models.SET_NULL, null=True, blank=True, related_name='findings', verbose_name="Associated Clause")
    rule = models.ForeignKey(RiskRule, on_delete=models.PROTECT, related_name='findings', verbose_name="Risk Rule")
    risk_score = models.DecimalField(max_digits=5, decimal_places=2, default=0.0, verbose_name="Risk Score")
    risk_level = models.CharField(max_length=20, choices=RISK_LEVEL_CHOICES, default='MEDIUM', verbose_name="Risk Level")
    explanation = models.TextField(verbose_name="Explanation")
    recommendation = models.TextField(blank=True, null=True, verbose_name="Recommendation")
    disadvantaged_party = models.CharField(max_length=255, blank=True, null=True, verbose_name="Disadvantaged Party")

    class Meta:
        verbose_name = 'Risk Finding'
        verbose_name_plural = 'Risk Findings'

    def __str__(self):
        return f"Finding: {self.rule.rule_name} ({self.risk_level})"


class Review(models.Model):
    analysis = models.ForeignKey(AIAnalysis, on_delete=models.CASCADE, related_name='reviews', verbose_name="AI Analysis")
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='reviews', verbose_name="Reviewer")
    note = models.TextField(verbose_name="Note")
    decision = models.CharField(max_length=20, verbose_name="Decision")
    reviewed_at = models.DateTimeField(auto_now_add=True, verbose_name="Reviewed At")

    class Meta:
        verbose_name = 'Expert Review'
        verbose_name_plural = 'Expert Reviews'

    def __str__(self):
        return f"Review by {self.user.username}"





class ContractParty(models.Model):
    contract = models.ForeignKey(Contract, on_delete=models.CASCADE, related_name='parties', verbose_name="Contract")
    party_name = models.CharField(max_length=255, verbose_name="Party Name")
    tax_code = models.CharField(max_length=100, blank=True, null=True, verbose_name="Tax Code")
    email = models.EmailField(blank=True, null=True, verbose_name="Email")
    phone = models.CharField(max_length=50, blank=True, null=True, verbose_name="Phone")
    party_type = models.CharField(max_length=100, verbose_name="Party Type")

    class Meta:
        verbose_name = 'Contract Party'
        verbose_name_plural = 'Contract Parties'

    def __str__(self):
        return self.party_name


class Notification(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='notifications', verbose_name="User")
    title = models.CharField(max_length=255, verbose_name="Title")
    message = models.TextField(verbose_name="Message")
    is_read = models.BooleanField(default=False, verbose_name="Is Read")

    class Meta:
        verbose_name = 'Notification'
        verbose_name_plural = 'Notifications'

    def __str__(self):
        return self.title


class AuditLog(models.Model):
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='audit_logs', verbose_name="User")
    contract = models.ForeignKey(Contract, on_delete=models.CASCADE, null=True, blank=True, related_name='audit_logs', verbose_name="Contract")
    action = models.CharField(max_length=100, verbose_name="Action")
    ip_address = models.CharField(max_length=45, blank=True, null=True, verbose_name="IP Address")
    payload = models.TextField(blank=True, null=True, verbose_name="Request Payload")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Created At")

    class Meta:
        verbose_name = 'Audit Log'
        verbose_name_plural = 'Audit Logs'

    def __str__(self):
        username = self.user.username if self.user else "System"
        return f"{username} - {self.action}"
