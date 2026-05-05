import ctypes
import logging

import numpy as np
import pycuda.driver as drv  # type: ignore
from pycuda import gpuarray  # type: ignore

from pydtnn.backends.pycuda.layers.layer import LayerPycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.layers.input import Input
from pydtnn.utils.constants import ArrayShape

__all__ = ("InputPycuda",)

logger = logging.getLogger(__name__)


class InputPycuda(Input[TensorArray], LayerPycuda):
    ws_size = 0
    ws: drv.DeviceAllocation = None

    def _model_init(self, prev_shape: ArrayShape, x: TensorArray):
        super()._model_init(prev_shape, x)

        y_gpu = gpuarray.zeros((self.model.batch_size, *self.shape), self.model.dtype)
        self.y = TensorArray(y_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        self.memory_used += self.y.nbytes

    def forward(self, x: TensorArray) -> TensorArray:
        return x

    def backward(self, dy: TensorArray) -> TensorArray:
        return dy

    def _sync_x_y(self, x_batch: np.ndarray, y_batch: np.ndarray) -> tuple[TensorArray, TensorArray]:
        # NOTE: in CUDA it's necessary to always have batches of the same size.
        local_batch_size = x_batch.shape[0]

        if local_batch_size != 0:
            if local_batch_size != self.model.batch_size:
                # NOTE: if x_batch is empty (local_batch_size == 0), this will mean the end of the loop where this function is called.
                num_repetitions = np.ceil(self.model.batch_size / local_batch_size)
                x_batch = np.repeat(x_batch, num_repetitions, axis=0)[: self.model.batch_size]
                y_batch = np.repeat(y_batch, num_repetitions, axis=0)[: self.model.batch_size]
            # else: The batch has the right shape ==> Nothing to do.

            x_batch = np.asarray(x_batch, dtype=self.model.dtype, order="C")
            y_batch = np.asarray(y_batch, dtype=self.model.dtype, order="C")

            assert isinstance(self.y, TensorArray) and isinstance(self.model.y_batch, TensorArray)
            self.y.set(x_batch)
            self.model.y_batch.set(y_batch)
            x, y_targ = self.model.layers[0].y, self.model.y_batch
        else:
            empty_x = gpuarray.zeros((1, *self.model.dataset.input_shape), self.model.dtype)[:0]
            empty_y_tag = gpuarray.zeros((1, *self.model.dataset.output_shape), self.model.dtype)[:0]
            x = TensorArray(empty_x, self.model.tensor_format, self.model.cudnn_dtype)
            y_targ = TensorArray(empty_y_tag, self.model.tensor_format, self.model.cudnn_dtype)
        return x, y_targ

    @property
    def ws_ptr(self) -> ctypes.c_void_p:
        return ctypes.c_void_p(int(self.ws))

    def checkConvolutionMemory(self, size) -> None:
        if size.value < self.ws_size:
            return

        if self.ws is not None:
            self.ws.free()

        self.ws_size = max(1, size.value)
        self.ws = drv.mem_alloc(self.ws_size)

    def getConvolutionWorkspacePtr(self) -> ctypes.c_void_p:
        return self.ws_ptr

    def getConvolutionWorkspaceSize(self) -> int:
        return self.ws_size
