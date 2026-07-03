"""Module providing the base class for NumPy-based backends in PyDTNN."""

import numpy as np

from pydtnn.abstract.base import Base

__all__ = ("BaseNumpy",)


class BaseNumpy(Base[np.ndarray]):
    """Abstract base class for all NumPy-backed components in the framework."""

    ...
