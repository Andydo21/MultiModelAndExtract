from django.contrib import admin
from .models import ShopfloorInquiry, InquiryMessage
from .models import ExpertCase, CaseAuditEvent


@admin.register(ExpertCase)
class ExpertCaseAdmin(admin.ModelAdmin):
    list_display = ('title', 'machine', 'revision', 'status', 'author', 'reviewer')
    list_filter = ('status', 'machine')
    search_fields = ('title', 'symptom', 'error_code')

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False


@admin.register(CaseAuditEvent)
class CaseAuditEventAdmin(admin.ModelAdmin):
    list_display = ('case', 'actor', 'action', 'created_at')

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False

@admin.register(ShopfloorInquiry)
class ShopfloorInquiryAdmin(admin.ModelAdmin):
    list_display = ('ticket_code', 'machine', 'operator_name', 'assigned_expert', 'status', 'neo4j_synced', 'created_at')
    list_filter = ('status', 'neo4j_synced', 'machine')
    search_fields = ('ticket_code', 'question', 'expert_answer', 'operator_name')

@admin.register(InquiryMessage)
class InquiryMessageAdmin(admin.ModelAdmin):
    list_display = ('inquiry', 'sender_role', 'sender_name', 'created_at')
    list_filter = ('sender_role',)
