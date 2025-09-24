"""Message Passing Interface"""

import sys as _sys

from pydtnn.libs.mpi import rc
from pydtnn import comms as _comms


# Select implementation
match rc.proto:
    case _comms.Protocol():
        from pydtnn.libs.mpi import client as _module

    case _:
        from mpi4py import MPI as _module


# Replace module
_sys.modules[__name__] = _module


def __getattr__(key):
    """Proxy all attributes to module"""
    return getattr(_module, key)
