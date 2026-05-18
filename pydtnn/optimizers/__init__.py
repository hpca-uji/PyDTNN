"""
PyDTNN optimizers module.

This module provides utilities for managing and retrieving optimizer implementations
within the PyDTNN framework.
"""

from pydtnn.optimizers.abstract.optimizer import Optimizer
from pydtnn.utils import find_component


def select(name: str) -> type[Optimizer]:
    """
    Selects an optimizer class by its name.

    Args:
        name (str): The name of the optimizer class to retrieve.

    Returns:
        type[Optimizer]: The requested optimizer class.
    """
    assert __package__, "Package not found!"
    return find_component(__package__, name)
