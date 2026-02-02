import numpy as np
import pycuda.gpuarray as gpuarray  # type: ignore

from pydtnn.layers.input import Input
from pydtnn.backends.gpu.layers.layer import LayerGPU
from pydtnn.backends.gpu.utils.tensor_gpu import TensorGPU
from pydtnn.utils.constants import ArrayShape


class InputGPU(Input[TensorGPU], LayerGPU):

    def _sync_x_y(self, x_batch: np.ndarray, y_batch: np.ndarray) -> tuple[TensorGPU, TensorGPU]:
        # NOTE: in CUDA it's necessary to always have batches of the same size.
        local_batch_size = x_batch.shape[0]

        if local_batch_size != 0:
            if local_batch_size != self.model.batch_size:
                # NOTE: if x_batch is empty (local_batch_size == 0), this will mean the end of the loop where this function is called.
                num_repetitions = ceil(self.model.batch_size / local_batch_size)
                x_batch = np.repeat(x_batch, num_repetitions, axis=0)[:self.model.batch_size]
                y_batch = np.repeat(y_batch, num_repetitions, axis=0)[:self.model.batch_size]
            # else: The batch has the right shape ==> Nothing to do.

            x_batch = np.asarray(x_batch, dtype=self.model.dtype, order='C', copy=None)
            y_batch = np.asarray(y_batch, dtype=self.model.dtype, order='C', copy=None)

            assert isinstance(self.y, TensorGPU) and isinstance(self.model.y_batch, TensorGPU)
            self.y.ary.set(x_batch)
            self.model.y_batch.ary.set(y_batch)
            x, y_targ = self.model.layers[0].y, self.model.y_batch
        else:
            empty_x = gpuarray.empty((1, *self.model.dataset.input_shape), self.model.dtype)[:0]
            empty_y_tag = gpuarray.empty((1, *self.model.dataset.output_shape), self.model.dtype)[:0]
            x = TensorGPU(empty_x, self.tensor_format, self.cudnn_dtype)
            y_targ = TensorGPU(empty_y_tag, self.model.tensor_format, self.model.cudnn_dtype)
        return x, y_targ

    def initialize(self, prev_shape: ArrayShape, x: TensorGPU):
        super().initialize(prev_shape, x)

        if not self.model.enable_cudnn:
            raise RuntimeError("GPU layers requires CUDNN to be enabled!")

        y_gpu = gpuarray.empty((self.model.batch_size, *self.shape), self.model.dtype)
        self.y = TensorGPU(y_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        self.real_memory_size += self.y.nbytes

    def forward(self, x: TensorGPU) -> TensorGPU:
        return x

    def backward(self, dy: TensorGPU) -> TensorGPU:
        return dy
