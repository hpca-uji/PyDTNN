import numpy as np

from pydtnn.abstract.base import Base


__all__ = (
    "BaseNumpy",
)


class BaseNumpy(Base[np.ndarray]):
    ...
