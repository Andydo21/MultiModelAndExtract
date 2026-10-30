from django.db import migrations


def permissions(apps, schema_editor):
    Permission = apps.get_model('contracts', 'Permission')
    Permission.objects.get_or_create(permission_code='ledger.sign', defaults={'description': 'Sign an exact contract version using an approved RSA key'})
    Permission.objects.get_or_create(permission_code='ledger.anchor', defaults={'description': 'Queue a version hash for Fabric anchoring'})


class Migration(migrations.Migration):
    dependencies = [('contracts', '0011_contractsigningcredential_contractledgeroperation')]
    operations = [migrations.RunPython(permissions, migrations.RunPython.noop)]
