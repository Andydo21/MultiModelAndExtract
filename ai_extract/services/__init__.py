"""Public domain classes; implementations are grouped by responsibility."""
from importlib import import_module

_public_services = {'SummarizeService', 'ExtractEntityService', 'ClauseExtractService'}


def __getattr__(name):
    if name in _public_services:
        return getattr(import_module(f'{__name__}.extraction_service'), name)
    raise AttributeError(name)
