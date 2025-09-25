"""Message Passing Interface"""

import os as _os
import sys as _sys


# Select implementation
if _os.environ.get("PYMPI_PROTO"):
    from pympi import MPI as _module
else:
    from mpi4py import MPI as _module

# Replace module
_sys.modules[__name__] = _module

if hasattr(_module, "__all__"):
    __all__ = _module.__all__  # type: ignore


def __getattr__(key):
    """Proxy all attributes to module"""
    return getattr(_module, key)
