"""Message Passing Interface"""

import sys as _sys
import typing as _typing

from pydtnn.libs.mpi import rc as _rc

# Select implementation
if _rc.proto:
    from pympi import MPI as _module  # noqa: N811
else:
    from mpi4py import MPI as _module  # noqa: N811

# Replace module
_sys.modules[__name__] = _module

if hasattr(_module, "__all__"):
    __all__ = _module.__all__  # pyright: ignore[reportAttributeAccessIssue,reportUnsupportedDunderAll]


def __getattr__(key: _typing.Any) -> _typing.Any:
    """Proxy all attributes to module"""
    return getattr(_module, key)
