"""Message Passing Interface (self-contained)"""

import sys as _sys
from mpi4py import MPI as _module


# Replace module
_sys.modules[__name__] = _module


def __getattr__(key):
    """Proxy all attributes to implementation"""
    return getattr(_module, key)
