"""Message Passing Interface."""

import importlib

_module = importlib.import_module("mpi4py.MPI")


def __getattr__(key):
    """Proxy all attributes to implementation"""
    return getattr(_module, key)
