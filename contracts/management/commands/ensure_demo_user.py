"""Create the admin user that AutoDemoLoginMiddleware attaches, so a new database opens without a login."""
from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand

from contracts.models import Company, Role


class Command(BaseCommand):
    help = 'Create a local admin user for login-free access, if the database has no superuser.'

    def handle(self, *args, **options):
        User = get_user_model()
        existing = User.objects.filter(is_superuser=True).first()
        if existing:
            self.stdout.write(f'Superuser {existing.username} already exists; nothing to do.')
            return
        company, _ = Company.objects.get_or_create(company_name='Demo', defaults=dict(tax_code='DEMO'))
        role, _ = Role.objects.get_or_create(role_name='ADMIN')
        user = User(username='admin', company=company, role=role, is_staff=True, is_superuser=True)
        user.set_unusable_password()
        user.save()
        self.stdout.write('Created superuser admin; pages now open without a login.')
