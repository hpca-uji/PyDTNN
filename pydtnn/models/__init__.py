"""
Module for model selection and component retrieval within the PyDTNN framework.
"""

from collections.abc import Callable, Sequence

from pydtnn.abstract.layerable import Layerable
from pydtnn.utils import find_component
from pydtnn.utils.constants import ArrayShape

__all__ = ("select",)


def select(name: str) -> Callable[[ArrayShape, ArrayShape], Sequence[Layerable]]:
    """
    Retrieve a model component factory by name.

    Args:
        name: The identifier of the model component to retrieve.

    Returns:
        A callable that constructs a sequence of Layerable objects.
    """
    assert __package__, "Package not found!"
    return find_component(__package__, name)
