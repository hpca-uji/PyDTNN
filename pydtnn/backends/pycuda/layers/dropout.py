import ctypes
import logging

import numpy as np
from pycuda import gpuarray  # type: ignore

from pydtnn.backends.pycuda.layers.layer import LayerPycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.layers.dropout import Dropout
from pydtnn.libs import cudnn as cudnn
from pydtnn.tracers.events import PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_OPS_EVENT_enum
from pydtnn.utils.constants import ArrayShape

__all__ = (
    "DropoutPycuda",
)

logger = logging.getLogger(__name__)


class DropoutPycuda(Dropout[TensorArray], LayerPycuda):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # The following values will be initalized later:
        self.states_size: ctypes.c_size_t = None  # type: ignore
        self.space_size: ctypes.c_size_t = None  # type: ignore
        self.space: TensorArray = None  # type: ignore
        self.states: TensorArray = None  # type: ignore
        self.drop_desc: int | None = None

    def _model_init(self, prev_shape: ArrayShape, x: TensorArray) -> None:
        super()._model_init(prev_shape, x)

        # Activations y
        y_gpu = gpuarray.zeros((self.model.batch_size, *self.shape), self.model.dtype)
        self.y = TensorArray(y_gpu, self.model.tensor_format, self.model.cudnn_dtype)
        self.memory_used += self.y.nbytes

        # Derivative dx
        dx_gpu = gpuarray.zeros((self.model.batch_size, *self.shape), self.model.dtype)
        self.dx = TensorArray(dx_gpu, self.model.tensor_format, self.model.cudnn_dtype)
        self.memory_used += self.dx.nbytes

        self.states_size = cudnn.cudnnDropoutGetStatesSize(self.model.cudnn_handle)
        self.space_size = cudnn.cudnnDropoutGetReserveSpaceSize(self.y.desc)

        space_gpu = gpuarray.zeros((self.space_size.value,), np.byte)
        self.space = TensorArray(space_gpu, self.model.tensor_format, self.model.cudnn_dtype, TensorArray.TensorType.OTHER)
        self.memory_used += self.space.nbytes

        states_gpu = gpuarray.zeros((self.states_size.value,), np.byte)
        self.states = TensorArray(states_gpu, self.model.tensor_format, self.model.cudnn_dtype, TensorArray.TensorType.OTHER)
        self.memory_used += self.states.nbytes

        self.drop_desc = cudnn.cudnnCreateDropoutDescriptor()

        cudnn.cudnnSetDropoutDescriptor(self.drop_desc, self.model.cudnn_handle, self.rate,
                                        self.states.ptr_voidp, self.states_size, seed=0)

    def forward(self, x: TensorArray) -> TensorArray:
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CUDNN)
        cudnn.cudnnDropoutForward(self.model.cudnn_handle, self.drop_desc,
                                  x.desc, x.ptr_voidp,
                                  self.y.desc, self.y.ptr_voidp,
                                  self.space.ptr_voidp, self.space_size.value)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return self.y

    def backward(self, dy: TensorArray) -> TensorArray:
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_CUDNN_DX)
        # Compute dx
        cudnn.cudnnDropoutBackward(self.model.cudnn_handle, self.drop_desc,
                                   dy.desc, dy.ptr_voidp,
                                   self.dx.desc, self.dx.ptr_voidp,
                                   self.space.ptr_voidp, self.space_size.value)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return self.dx
