"""Module providing the base class for NumPy-based backends in PyDTNN."""

from typing import TYPE_CHECKING

from pydtnn.libs import numpy as np
from pydtnn.abstract.base import Base

__all__ = ("BaseNumpy",)

if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)


class BaseNumpy(Base[np.ndarray]):
    """Abstract base class for all NumPy-backed components in the framework."""
