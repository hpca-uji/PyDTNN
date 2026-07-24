"""Constants and type definitions for the PyDTNN framework."""

import logging
import typing
from enum import StrEnum, auto

import numpy as np

__all__ = ("Components", "NetworkAlgoEnum", "Parameters", "ArrayShape", "DTYPE2CTYPE", "Array")

logger = logging.getLogger(__name__)


type ArrayShape = tuple[int, ...]
DTYPE2CTYPE: dict[np.dtype, str] = {np.dtype(np.float32): "float", np.dtype(np.float64): "double"}


class NetworkAlgoEnum(StrEnum):
    """Enumeration of supported network algorithms."""

    BTA = auto()
    VDG = auto()


class Components(StrEnum):
    """Enumeration of core framework component categories."""

    DATASETS = auto()
    ACTIVATIONS = auto()
    LAYERS = auto()
    LOSSES = auto()
    METRICS = auto()
    MODELS = auto()
    OPTIMIZERS = auto()
    SCHEDULERS = auto()
    TRACERS = auto()


class Parameters(StrEnum):
    """Enumeration of parameter keys used in model state and gradients."""

    PATHS = auto()
    CANONICAL_NAME = auto()
    MODEL_NAME = auto()
    LAYERS = auto()

    RUNNING_MEAN = auto()
    RUNNING_VAR = auto()
    WEIGHTS = auto()
    DW = auto()
    BIASES = auto()
    DB = auto()


# NOTE: It is necessary to have "ArrayShape" initialized before TensorGPU
if typing.TYPE_CHECKING:
    from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
type Array = "np.ndarray | TensorArray"
