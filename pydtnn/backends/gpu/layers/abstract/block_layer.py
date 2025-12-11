from pydtnn.backends.gpu.layers.layer import LayerGPU
from pydtnn.backends.gpu.utils.tensor_gpu import TensorGPU
from pydtnn.layers.abstract.block_layer import AbstractBlockLayer


class AbstractBlockLayerGPU(AbstractBlockLayer[TensorGPU], LayerGPU):
    pass
