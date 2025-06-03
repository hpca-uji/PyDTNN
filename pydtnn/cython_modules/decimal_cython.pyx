"""Decimal utilities"""

from libc.math cimport roundf

__all__ = (
    "round",
)

def round(x: float) -> int:
    return <int>roundf(x)
