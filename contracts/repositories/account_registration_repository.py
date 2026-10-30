"""Database operations for account registration."""
from contracts.models import Company, Role, User

def exists_user_for_register_account(username):
    return User.objects.filter(username=username).exists()


def email_already_registered(email):
    return User.objects.filter(email=email).exists()


def get_company_for_register_account(company_id):
    return Company.objects.get(id=company_id)


def get_role_for_register_account(role_id):
    return Role.objects.get(id=role_id)


def create_user_user_for_register_account(company, email, password, role, username):
    return User.objects.create_user(username=username, email=email, password=password, company=company, role=role, status='ACTIVE')


def save_record_for_register_account(user):
    return user.save()


def get_or_create_role_for_ensure_manager_role():
    return Role.objects.get_or_create(role_name='MANAGER')


def order_by_company_for_registration_context():
    return Company.objects.all().order_by('company_name')


def order_by_role_for_registration_context():
    return Role.objects.all().order_by('role_name')

