"""E2: A–D frozen graph expansion with traceable scope/token rejection."""
from .common import execute


def run(root,config,out,**options):
    return execute(root,config,out,'E2',**options)
