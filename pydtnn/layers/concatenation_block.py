from abc import ABC

from pydtnn.layers.abstract_block_layer import AbstractBlockLayer

from pydtnn.utils.tensor import decode_tensor, PYDTNN_TENSOR_FORMAT
from pydtnn.layers.layer import LayerError
from pydtnn.utils.types import shape_t

import numpy as np

CONCAT_DIM_NCHW = 1
CONCAT_DIM_NHWC = -1

class ConcatenationBlock(AbstractBlockLayer, ABC):

    def show(self, attrs="") -> None:
        print(
            f"|{self.id:^7d}"
            f"|{(type(self).__name__.replace('Concatenation', 'Concat') + ' (%d-path)' % len(self.paths)):^26s}"
            f"|{'':9s}|{str(self.shape):^15s}|{'':19s}|{'':37s}|")
        for i, p in enumerate(self.paths):
            print(f"|{('Path %d' % i):^7s}|{'':^26s}|{'':9s}|{'':15s}|{'':19s}|{'':37s}|")
            for layer in p:
                layer.show()

    def initialize_block_layer(self):
        super().initialize_block_layer(self)

        match self.model.tensor_format:
            case PYDTNN_TENSOR_FORMAT.NCHW:
                self.concat_dim = CONCAT_DIM_NCHW
                if not all([tuple(o[CONCAT_DIM_NCHW:]) == tuple(self.out_shapes[0][CONCAT_DIM_NCHW:]) for o in self.out_shapes]):
                    raise LayerError(f"All output shape must have the same number of elements.\n{self.out_shapes}")
                self.out_co = [s[0] for s in self.out_shapes]
                self.idx_co = np.cumsum(self.out_co, axis=0)
                self.shape = (sum(self.out_co), *self.out_shapes[0][CONCAT_DIM_NCHW:])
            case PYDTNN_TENSOR_FORMAT.NHWC:
                self.concat_dim = CONCAT_DIM_NHWC
                if not all([tuple(o[:CONCAT_DIM_NHWC]) == tuple(self.out_shapes[0][:CONCAT_DIM_NHWC]) for o in self.out_shapes]):
                    raise LayerError(f"All output shape must have the same number of elements.\n{self.out_shapes}")
                self.out_co = [s[-1] for s in self.out_shapes]
                self.idx_co: np.ndarray = np.cumsum(self.out_co, axis=0)
                self.shape: shape_t = (*self.out_shapes[0][:CONCAT_DIM_NHWC], sum(self.out_co))
            case _:
                raise NotImplementedError(f"\"ConcatenationBlock\" is not implemented for \"{self.model.tensor_format}\" format.")
        self.ho, self.wo, self.co = decode_tensor(self.shape, self.model.tensor_format)

