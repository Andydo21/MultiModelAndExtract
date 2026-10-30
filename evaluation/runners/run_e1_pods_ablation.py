"""E1: A–E evidence representation with fixed evidence/order/generator."""
from .common import execute


def run(root,config,out,**options):
    return execute(root,config,out,'E1',**options)
