"""Database operations for identity workflows."""
from contracts.models import Company
from contracts.models import User
from contracts.models import Role
from contracts.models import Company, Role, User
from contracts.models import User

def find_company_by_tax_code(tax_code):
    return Company.objects.filter(tax_code=tax_code).first()


def find_company_by_name(name):
    return Company.objects.filter(company_name=name).first()


def create_company_for_register_company(name, tax_code):
    return Company.objects.create(company_name=name, tax_code=tax_code, status='ACTIVE')


def save_company_for_register_company(company):
    return company.save()


def get_company_for_register_user(company_id):
    return Company.objects.get(id=company_id)


def get_role_for_register_user(role_id):
    return Role.objects.get(id=role_id)


def first_user_for_register_user(username):
    return User.objects.filter(username=username).first()


def save_user_for_register_user(user):
    return user.save()


def order_by_company_for_companies_list():
    return Company.objects.all().order_by('company_name', '-id')


def order_by_user_for_users_list():
    return User.objects.all().select_related('company', 'role').order_by('-id')


def get_or_create_role_for_roles_list():
    return Role.objects.get_or_create(role_name='MANAGER')


def order_by_role_for_roles_list():
    return Role.objects.all().order_by('role_name')


def create_registered_user(username, email, password, company, role):
    return User.objects.create_user(username=username, email=email, password=password,
        company=company, role=role, status='ACTIVE')
