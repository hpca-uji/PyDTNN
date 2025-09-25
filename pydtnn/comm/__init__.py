"""Message Passing Interface"""

import os as _os
import pympi.rc as _rc
import net_queue as _nq

__all__ = _rc.__all__  # type: ignore


# Redefine protocol
proto = (
    _nq.Protocol(proto)
    if (proto := _os.environ.get("PYMPI_PROTO"))
    else None
)


def __getattr__(key):
    """Proxy all attributes to module"""
    return getattr(_rc, key)
