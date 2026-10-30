import json
import uuid
from django.db import models


class ExpertCase(models.Model):
    """Reviewed, revisioned A2 cases separate from legacy unmanaged tables."""
    case_id = models.UUIDField(default=uuid.uuid4, editable=False, db_index=True)
    revision = models.PositiveIntegerField(default=1)
    title = models.CharField(max_length=255)
    machine = models.CharField(max_length=128, db_index=True)
    component = models.CharField(max_length=128, blank=True)
    error_code = models.CharField(max_length=64, blank=True)
    symptom = models.TextField()
    investigation = models.TextField(blank=True)
    root_cause = models.TextField(blank=True)
    solution = models.TextField(blank=True)
    verification = models.TextField(blank=True)
    applicability = models.TextField(blank=True)
    source_reference = models.CharField(max_length=255, blank=True)
    author = models.ForeignKey('auth.User', on_delete=models.PROTECT, related_name='a2_cases')
    reviewer = models.ForeignKey('auth.User', null=True, blank=True, on_delete=models.PROTECT,
                                 related_name='reviewed_a2_cases')
    status = models.CharField(max_length=20, default='draft',
                              choices=[('draft', 'Nháp'), ('approved', 'Đã duyệt'),
                                       ('deprecated', 'Ngừng sử dụng')], db_index=True)
    approved_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']
        constraints = [models.UniqueConstraint(fields=['case_id', 'revision'],
                                                name='unique_a2_case_revision')]

    def __str__(self):
        return f'{self.title} (r{self.revision}, {self.status})'


class CaseAuditEvent(models.Model):
    case = models.ForeignKey(ExpertCase, on_delete=models.PROTECT, related_name='events')
    actor = models.ForeignKey('auth.User', on_delete=models.PROTECT)
    action = models.CharField(max_length=32)
    created_at = models.DateTimeField(auto_now_add=True)

# ==============================================================================
# EXISTING DENSO FACTORY MEMORY MODELS (Mapped directly to SQLite database)
# ==============================================================================

class Document(models.Model):
    doc_code = models.CharField(max_length=64, unique=True, default='DOC001')
    title = models.CharField(max_length=255)
    doc_type = models.CharField(max_length=20, default='SOP')
    description = models.TextField(blank=True, null=True)
    file_path = models.FileField(upload_to='documents/%Y/%m/', blank=True, null=True)
    language = models.CharField(max_length=64, default='Japanese/English/Vietnamese')
    status = models.CharField(max_length=30, default='PROCESSED')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        managed = False
        db_table = 'core_document'
        ordering = ['-created_at']

    def __str__(self):
        return f"[{self.doc_code}] {self.title}"


class DocumentPage(models.Model):
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='pages')
    page_number = models.IntegerField()
    image_path = models.ImageField(upload_to='pages/%Y/%m/', blank=True, null=True)
    ocr_text = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        managed = False
        db_table = 'core_documentpage'
        ordering = ['page_number']

    def __str__(self):
        return f"{self.document.doc_code} — Trang {self.page_number}"


class Chunk(models.Model):
    page = models.ForeignKey(DocumentPage, on_delete=models.CASCADE, related_name='chunks')
    content = models.TextField()
    chunk_type = models.CharField(max_length=20, default='text')
    bbox_json = models.TextField(default='[]')
    embedding_json = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        managed = False
        db_table = 'core_chunk'

    def get_bbox(self):
        try:
            return json.loads(self.bbox_json)
        except Exception:
            return []

    def __str__(self):
        return f"Chunk {self.id} ({self.chunk_type}) - Trang {self.page.page_number}"


class Expert(models.Model):
    expert_code = models.CharField(max_length=64, unique=True, default='EXP001')
    name = models.CharField(max_length=128)
    department = models.CharField(max_length=128, default='Bảo trì Cơ điện (Maintenance)')
    position = models.CharField(max_length=128, default='Senior Line Master')
    experience_years = models.IntegerField(default=10)
    avatar_url = models.CharField(max_length=255, blank=True, null=True)
    description = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        managed = False
        db_table = 'core_expert'

    def __str__(self):
        return f"{self.name} ({self.position})"


class ExpertSkill(models.Model):
    expert = models.ForeignKey(Expert, on_delete=models.CASCADE, related_name='skills')
    skill_name = models.CharField(max_length=128)
    level = models.IntegerField(default=5)

    class Meta:
        managed = False
        db_table = 'core_expertskill'


class Case(models.Model):
    case_code = models.CharField(max_length=64, unique=True, default='CASE-101')
    title = models.CharField(max_length=255)
    machine = models.CharField(max_length=128)
    part_code = models.CharField(max_length=64, default='N/A')
    symptom = models.TextField()
    root_cause = models.TextField()
    solution = models.TextField()
    result = models.CharField(max_length=64, default='Resolved')
    expert = models.ForeignKey(Expert, on_delete=models.SET_NULL, null=True, blank=True, related_name='cases')
    embedding_json = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        managed = False
        db_table = 'core_case'
        ordering = ['-created_at']

    def __str__(self):
        return f"[{self.case_code}] {self.title} ({self.machine})"


class Knowledge(models.Model):
    knowledge_code = models.CharField(max_length=64, unique=True, default='K001')
    title = models.CharField(max_length=255)
    content = models.TextField()
    knowledge_type = models.CharField(max_length=20, default='RULE')
    source_type = models.CharField(max_length=50, default='EXPERT')
    source_id = models.CharField(max_length=64, default='EXP001')
    status = models.CharField(max_length=20, default='APPROVED')
    confidence = models.FloatField(default=0.92)
    approved_by = models.ForeignKey(Expert, on_delete=models.SET_NULL, null=True, blank=True, related_name='approved_rules')
    embedding_json = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        managed = False
        db_table = 'core_knowledge'
        ordering = ['-created_at']

    def __str__(self):
        return f"[{self.knowledge_code}] {self.title}"


# ==============================================================================
# NEWIDEA EXCLUSIVE: UNIFIED CHAT, EXPERT ROUTING & ZERO-ERROR PIPELINE MODELS
# ==============================================================================

class ShopfloorInquiry(models.Model):
    """
    Thắc mắc hiện trường từ Nhân viên / Kỹ thuật viên khi gặp sự cố máy móc:
    - Khi AI chưa đủ căn cứ (Layer 4 Fail-Safe), hệ thống tự động chỉ định Chuyên gia phụ trách.
    - Chuyên gia nhận thông báo chuông đỏ, mở khung chat chuyên dụng để giải đáp.
    """
    STATUS_CHOICES = (
        ('PENDING', 'Chờ Chuyên gia giải đáp (Pending Expert)'),
        ('IN_PROGRESS', 'Chuyên gia đang xử lý (In Progress)'),
        ('RESOLVED', 'Đã giải quyết & Lưu Neo4j (Resolved & Synced)'),
    )

    ticket_code = models.CharField(max_length=64, unique=True, default='INQ-001', verbose_name="Mã thắc mắc")
    operator_name = models.CharField(max_length=128, default='Kỹ thuật viên Hiện trường', verbose_name="Người hỏi")
    machine = models.CharField(max_length=128, default='Máy gia công M12', verbose_name="Thiết bị / Cụm máy")
    question = models.TextField(verbose_name="Nội dung câu hỏi thắc mắc")
    
    assigned_expert = models.ForeignKey(Expert, on_delete=models.SET_NULL, null=True, blank=True, related_name='assigned_inquiries', verbose_name="Chuyên gia được chỉ định")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING', verbose_name="Trạng thái")
    
    ai_preliminary_analysis = models.TextField(blank=True, null=True, verbose_name="Phân tích sơ bộ từ AI")
    confidence_score = models.FloatField(default=0.0, verbose_name="Điểm tin cậy AI")
    
    expert_answer = models.TextField(blank=True, null=True, verbose_name="Câu trả lời chính thức của Chuyên gia")
    extracted_entities_json = models.TextField(default='{}', verbose_name="Thực thể bóc tách (Machine, Component, Symptom, RootCause, Solution)")
    cypher_executed = models.TextField(blank=True, null=True, verbose_name="Câu lệnh Cypher đã thực thi vào Neo4j")
    neo4j_synced = models.BooleanField(default=False, verbose_name="Đã nạp vào Neo4j Graph")
    
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Thời điểm gửi")
    resolved_at = models.DateTimeField(blank=True, null=True, verbose_name="Thời điểm chuyên gia giải đáp")

    class Meta:
        db_table = 'newidea_shopfloor_inquiry'
        ordering = ['-created_at']

    def __str__(self):
        return f"[{self.ticket_code}] {self.machine} - {self.question[:40]}... ({self.status})"

    def get_extracted_entities(self):
        try:
            return json.loads(self.extracted_entities_json)
        except Exception:
            return {}


class InquiryMessage(models.Model):
    """Lịch sử hội thoại trong từng phiên xử lý thắc mắc"""
    SENDER_ROLE_CHOICES = (
        ('operator', 'Kỹ thuật viên Hiện trường'),
        ('ai', 'Hệ thống AI Copilot'),
        ('expert', 'Chuyên gia Line Master'),
    )

    inquiry = models.ForeignKey(ShopfloorInquiry, on_delete=models.CASCADE, related_name='messages', null=True, blank=True)
    sender_role = models.CharField(max_length=20, choices=SENDER_ROLE_CHOICES, default='operator')
    sender_name = models.CharField(max_length=128, default='Kỹ thuật viên')
    message = models.TextField()
    metadata_json = models.TextField(default='{}')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'newidea_inquiry_message'
        ordering = ['created_at']

    def __str__(self):
        return f"[{self.sender_role.upper()}] {self.sender_name}: {self.message[:40]}"

    def get_metadata(self):
        try:
            return json.loads(self.metadata_json)
        except Exception:
            return {}
