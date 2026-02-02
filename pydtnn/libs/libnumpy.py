"""Numpy module"""

import os as _os
import sys as _sys


# Select implementation
if _os.environ.get("PYDTNN_CUPY"):
    import cupy as _module
else:
    import numpy as _module

# Replace module
_sys.modules[__name__] = _module

if hasattr(_module, "__all__"):
    __all__ = _module.__all__  # type: ignore


def __getattr__(key):
    """Proxy all attributes to module"""
    return getattr(_module, key)
