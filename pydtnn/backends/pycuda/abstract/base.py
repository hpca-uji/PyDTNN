from pydtnn.abstract.base import Base
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray


class BasePycuda(Base[TensorArray]):
    ...
