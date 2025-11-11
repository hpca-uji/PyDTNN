import typing
from enum import auto, StrEnum

import numpy as np


type ArrayShape = tuple[int, ...]
DTYPE2CTYPE: dict[np.dtype, str] = {
    np.dtype(np.float32): "float",
    np.dtype(np.float64): "double"
}


class NetworkAlgEnum(StrEnum):
    BTA = auto()
    VDG = auto()


class Components(StrEnum):
    DATASETS = auto()
    ACTIVATIONS = auto()
    LAYERS = auto()
    LOSSES = auto()
    METRICS = auto()
    MODELS = auto()
    OPTIMIZERS = auto()
    SCHEDULERS = auto()
    TRACERS = auto()


# NOTE: It is necessary to have "ArrayShape" initialized before TensorGPU
if typing.TYPE_CHECKING:
    from pydtnn.backends.gpu.utils.tensor_gpu import TensorGPU
type Array = "np.ndarray | TensorGPU"
