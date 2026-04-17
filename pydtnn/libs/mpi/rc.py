"""Message Passing Interface"""

import os as _os


try:
    import pympi.rc as _rc
except Exception:
    _rc = None

if _rc:
    __all__ = _rc.__all__  # type: ignore


# Redefine backend
proto = (
    proto
    if (proto := _os.environ.get("PYMPI_PROTO"))
    else None
)


def __getattr__(key):
    """Proxy all attributes to module"""
    return getattr(_rc, key)
