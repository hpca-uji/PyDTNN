"""Message Passing Interface"""

import os as _os
import sys as _sys
from pydtnn.libs.mpi import rc as _rc

# Select implementation
if _rc.proto:
    from pympi import MPI as _module  # type: ignore
else:
    from mpi4py import MPI as _module  # type: ignore

# Replace module
_sys.modules[__name__] = _module

if hasattr(_module, "__all__"):
    __all__ = _module.__all__  # type: ignore


def __getattr__(key):
    """Proxy all attributes to module"""
    return getattr(_module, key)
