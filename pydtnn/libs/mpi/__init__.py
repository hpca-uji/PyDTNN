"""Message Passing Interface"""

import os as _os

try:
    import pympi.rc as _rc
    import net_queue as _nq
except Exception:
    _rc = None
    _nq = None

if _rc:
    __all__ = _rc.__all__  # type: ignore


# Redefine backend
comm = (
    _nq.Backend(comm)
    if _nq and (comm := _os.environ.get("PYMPI_COMM"))
    else None
)


def __getattr__(key):
    """Proxy all attributes to module"""
    return getattr(_rc, key)
