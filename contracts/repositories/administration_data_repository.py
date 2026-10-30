"""Database operations for administration workflows."""
from contracts.models import User, Company, Contract, AuditLog
from contracts.models import User
from contracts.models import AuditLog
from contracts.models import User

def count_user_for_admin_stats():
    return User.objects.count()


def count_company_for_admin_stats():
    return Company.objects.count()


def count_contract_for_admin_stats():
    return Contract.objects.count()


def count_audit_log_for_admin_stats():
    return AuditLog.objects.count()


def order_by_audit_log_for_admin_stats():
    return AuditLog.objects.select_related('user', 'contract').order_by('-created_at')


def order_by_user_for_admin_stats():
    return User.objects.select_related('company', 'role').all().order_by('-id')


def get_user_for_admin_delete_user(user_id):
    return User.objects.get(id=user_id)


def delete_user_for_admin_delete_user(user):
    return user.delete()


def save_user_for_admin_retry_user(user):
    return user.save()


def create_audit_log_for_admin_toggle_ocr(metadata, skip_ocr, user):
    return AuditLog.objects.create(user=user, action=f'Changed system OCR configuration: SKIP_OCR={skip_ocr}', ip_address=metadata.get('REMOTE_ADDR'))
