from pydtnn.abstract.layerable import Layerable
from pydtnn.backends.pycuda.abstract.base import BasePycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray


class LayerablePycuda(Layerable[TensorArray], BasePycuda):
    ...
