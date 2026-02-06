from pydtnn.backends.pycuda.layers.layer import LayerPycuda
from pydtnn.backends.pycuda.utils.tensor_gpu import TensorGPU
from pydtnn.layers.abstract.block_layer import AbstractBlockLayer


class AbstractBlockLayerPycuda(AbstractBlockLayer[TensorGPU], LayerPycuda):
    pass
