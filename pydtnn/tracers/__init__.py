"""
PyDTNN tracer module.

This module provides utilities for managing and retrieving tracers
implementations within the PyDTNN framework.
"""

from pydtnn.tracers.abstract.base import Tracer
from pydtnn.utils import find_component


def select(name: str) -> type[Tracer]:
    """
    Select and return a tracer class by name.

    Args:
        name: The name of the tracer class to retrieve.

    Returns:
        The tracer class.
    """
    assert __package__, "Package not found!"
    return find_component(__package__, name)
