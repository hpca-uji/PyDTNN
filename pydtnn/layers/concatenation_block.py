from pydtnn.layers.abstract.block_layer import AbstractBlockLayer
from pydtnn.layers.layer import LayerError
from pydtnn.utils.tensor import TensorFormat
from pydtnn.utils.constants import ArrayShape, Array

import numpy as np

CONCAT_DIM_NCHW = 1
CONCAT_DIM_NHWC = -1

class ConcatenationBlock[T: Array](AbstractBlockLayer[T]):
    def initialize_block_layer(self):
        super().initialize_block_layer()

        match self.model.tensor_format:
            case TensorFormat.NCHW:
                self.concat_dim = CONCAT_DIM_NCHW
                if not all([tuple(o[CONCAT_DIM_NCHW:]) == tuple(self.out_shapes[0][CONCAT_DIM_NCHW:]) for o in self.out_shapes]):
                    raise LayerError(f"All output shape must have the same number of elements.\n{self.out_shapes}")
                self.out_co = [s[0] for s in self.out_shapes]
                self.idx_co = np.cumsum(self.out_co, axis=0)
                self.shape = (sum(self.out_co), *self.out_shapes[0][CONCAT_DIM_NCHW:])
            case TensorFormat.NHWC:
                self.concat_dim = CONCAT_DIM_NHWC
                if not all([tuple(o[:CONCAT_DIM_NHWC]) == tuple(self.out_shapes[0][:CONCAT_DIM_NHWC]) for o in self.out_shapes]):
                    raise LayerError(f"All output shape must have the same number of elements.\n{self.out_shapes}")
                self.out_co = [s[-1] for s in self.out_shapes]
                self.idx_co: np.ndarray = np.cumsum(self.out_co, axis=0)
                self.shape: ArrayShape = (*self.out_shapes[0][:CONCAT_DIM_NHWC], sum(self.out_co))
            case tensor_format:
                raise NotImplementedError(f"\"ConcatenationBlock\" is not implemented for \"{tensor_format}\" format.")
        self.co, self.ho, self.wo = self.model.decode_shape(self.shape)
