"""Message Passing Interface"""

import importlib as _importlib
from pydtnn.comms import PROTOCOL as _PROTOCOL

try:  # Specific implementation
    _module = _importlib.import_module(f"pydtnn.comms.{_PROTOCOL}.MPI")
except ModuleNotFoundError:  # Client implementation
    from pydtnn.comms import mpi_client as _module


def __getattr__(key):
    """Proxy all attributes to implementation"""
    return getattr(_module, key)
