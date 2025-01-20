"""Message Passing Interface."""

import os
import importlib

_implementation = os.environ.get("PYDTNN_MPI", "mpi")
_module = importlib.import_module(f"pydtnn.comms.{_implementation}.MPI")


def __getattr__(key):
    """Proxy all attributes to implementation"""
    return getattr(_module, key)
