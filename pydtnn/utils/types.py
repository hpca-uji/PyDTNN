import numpy as np
from pydtnn.backends.gpu.tensor_gpu import TensorGPU
from enum import auto, StrEnum

type Array = np.ndarray | TensorGPU

class NetworkAlgEnum(StrEnum):
    BTA = auto()
    VDG = auto()

