"""Message Passing Interface"""

import os as _os
import typing as _typing

try:
    import pympi.rc as _rc
except Exception:
    _rc = None

if _rc:
    __all__ = _rc.__all__  # pyright: ignore[reportUnsupportedDunderAll]


# Redefine backend
proto = proto if (proto := _os.environ.get("PYMPI_PROTO")) else None


def __getattr__(key: _typing.Any) -> _typing.Any:
    """Proxy all attributes to module"""
    return getattr(_rc, key)
