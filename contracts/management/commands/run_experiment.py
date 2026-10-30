"""Django entry point for the independent evaluation package; no database use."""
from django.core.management.base import BaseCommand,CommandError

from evaluation.cli import add_arguments,dispatch


class Command(BaseCommand):
    help='Audit E0 and run frozen-evidence E1/E2 experiments with traceable raw artifacts.'
    requires_system_checks=[]
    requires_migrations_checks=False

    def add_arguments(self,parser):
        add_arguments(parser)

    def handle(self,*args,**options):
        try:
            path=dispatch(options)
        except (ValueError,KeyError,OSError) as error:
            raise CommandError(str(error)) from error
        self.stdout.write(str(path))
