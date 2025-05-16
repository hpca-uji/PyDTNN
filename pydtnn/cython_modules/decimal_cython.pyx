"""Decimal utilities"""

from libc.math cimport roundf

__all__ = (
    "round",
)

def round(x: float) -> float:
    return roundf(x)
