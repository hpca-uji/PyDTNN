import numpy as np
import typing
from enum import auto, StrEnum

if typing.TYPE_CHECKING:
    import pycuda.gpuarray as gpuarray  # type: ignore


type ArrayShape = tuple[int, ...]
DTYPE2CTYPE: dict[np.dtype, str] = {
    np.dtype(np.float32): "float",
    np.dtype(np.float64): "double"
}


class NetworkAlgEnum(StrEnum):
    BTA = auto()
    VDG = auto()


# NOTE: It is necessary to have "ArrayShape" initialized before TensorGPU
from pydtnn.backends.gpu.tensor_gpu import TensorGPU
type Array = np.ndarray | TensorGPU
