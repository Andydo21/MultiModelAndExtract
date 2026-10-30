from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import (
    User, Contract, ContractVersion, ContractFile, RiskRule, 
    AIAnalysis, RiskFinding, Review, AuditLog, Company, Permission, 
    Role, Tag, ContractContext, ContextEmbedding, 
    ContractParty, Notification
)
from ai_extract.models import Clause as AIClause, ExtractedEntity as AIExtractedEntity

@admin.register(User)
class UserAdmin(BaseUserAdmin):
    list_display = ('username', 'email', 'role', 'company', 'is_staff', 'is_active')
    list_filter = ('role', 'company', 'is_staff', 'is_active')
    fieldsets = BaseUserAdmin.fieldsets + (
        ('Custom Roles & Company', {'fields': ('role', 'company', 'status', 'password_hash')}),
    )
    add_fieldsets = BaseUserAdmin.add_fieldsets + (
        ('Custom Roles & Company', {'fields': ('role', 'company', 'status', 'password_hash')}),
    )


class ContractFileInline(admin.TabularInline):
    model = ContractFile
    extra = 1


class ClauseInline(admin.TabularInline):
    model = AIClause
    extra = 1


@admin.register(ContractVersion)
class ContractVersionAdmin(admin.ModelAdmin):
    list_display = ('contract', 'version_number', 'file_hash', 'created_at')
    list_filter = ('version_number', 'created_at')
    inlines = [ContractFileInline, ClauseInline]


@admin.register(Contract)
class ContractAdmin(admin.ModelAdmin):
    list_display = ('contract_code', 'title', 'contract_type', 'status', 'contract_value', 'company', 'created_at')
    list_filter = ('status', 'contract_type', 'company', 'created_at')
    search_fields = ('contract_code', 'title', 'contract_type')
    filter_horizontal = ('tags',)



@admin.register(RiskRule)
class RiskRuleAdmin(admin.ModelAdmin):
    list_display = ('rule_name', 'severity')
    list_filter = ('severity',)
    search_fields = ('rule_name', 'description')


class RiskFindingInline(admin.TabularInline):
    model = RiskFinding
    extra = 1


@admin.register(AIAnalysis)
class AIAnalysisAdmin(admin.ModelAdmin):
    list_display = ('version', 'model_name', 'overall_score', 'created_at')
    list_filter = ('model_name', 'created_at')
    search_fields = ('version__contract__contract_code', 'version__contract__title')
    inlines = [RiskFindingInline]


@admin.register(RiskFinding)
class RiskFindingAdmin(admin.ModelAdmin):
    list_display = ('analysis', 'rule', 'risk_level')
    list_filter = ('risk_level', 'rule')
    search_fields = ('explanation', 'recommendation')


@admin.register(Review)
class ReviewAdmin(admin.ModelAdmin):
    list_display = ('analysis', 'user', 'decision', 'reviewed_at')
    list_filter = ('decision', 'reviewed_at')
    search_fields = ('note',)


@admin.register(AuditLog)
class AuditLogAdmin(admin.ModelAdmin):
    list_display = ('user', 'action', 'contract', 'created_at')
    list_filter = ('action', 'created_at')
    search_fields = ('user__username', 'action')
    readonly_fields = ('user', 'action', 'contract', 'created_at')


# Register all new models
@admin.register(Company)
class CompanyAdmin(admin.ModelAdmin):
    list_display = ('company_name', 'tax_code', 'status')
    list_filter = ('status',)


@admin.register(Permission)
class PermissionAdmin(admin.ModelAdmin):
    list_display = ('permission_code', 'description')


@admin.register(Role)
class RoleAdmin(admin.ModelAdmin):
    list_display = ('role_name', 'description')


@admin.register(Tag)
class TagAdmin(admin.ModelAdmin):
    list_display = ('tag_name',)


@admin.register(ContractContext)
class ContractContextAdmin(admin.ModelAdmin):
    list_display = ('version', 'context_type', 'source', 'relevance_score')
    list_filter = ('context_type',)


@admin.register(ContextEmbedding)
class ContextEmbeddingAdmin(admin.ModelAdmin):
    list_display = ('context', 'vector_id', 'embedding_model')






@admin.register(ContractParty)
class ContractPartyAdmin(admin.ModelAdmin):
    list_display = ('contract', 'party_name', 'party_type', 'tax_code')
    list_filter = ('party_type',)


@admin.register(Notification)
class NotificationAdmin(admin.ModelAdmin):
    list_display = ('user', 'title', 'is_read')
    list_filter = ('is_read',)
