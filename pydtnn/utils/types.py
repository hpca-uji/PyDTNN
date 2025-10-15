import numpy as np
from enum import auto, StrEnum

type shape_t = tuple[int, ...]
GPU_SUPPORTED_TYPES = dict[np.number, str]({np.float32: "float", np.float64: "double"})

class NetworkAlgEnum(StrEnum):
    BTA = auto()
    VDG = auto()
# -------------------


# NOTE: It is necessary to have "shape_t" initialized before TensorGPU
from pydtnn.backends.gpu.tensor_gpu import TensorGPU
type Array = np.ndarray | TensorGPU
