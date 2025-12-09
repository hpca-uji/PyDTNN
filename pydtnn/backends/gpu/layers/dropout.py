import numpy as np
import pycuda.gpuarray as gpuarray  # type: ignore

from pydtnn.layers.dropout import Dropout
from pydtnn.tracers.events import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum
from pydtnn.backends.gpu.layers.layer import LayerGPU
from pydtnn.libs import libcudnn as cudnn
from pydtnn.backends.gpu.utils.tensor_gpu import TensorGPU
from pydtnn.utils.constants import ArrayShape
import ctypes


class DropoutGPU(Dropout[TensorGPU], LayerGPU):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # The following values will be initalized later:
        self.states_size: ctypes.c_size_t = None  #type: ignore
        self.space_size: ctypes.c_size_t = None  #type: ignore
        self.space: TensorGPU = None  #type: ignore
        self.states: TensorGPU = None  #type: ignore
        self.drop_desc: int | None = None
    # ----

    def initialize(self, prev_shape: ArrayShape, x: TensorGPU) -> None:
        super().initialize(prev_shape, x)

        # Activations y
        y_gpu = gpuarray.empty((self.model.batch_size, *self.shape), self.model.dtype)
        self.y = TensorGPU(y_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        # Derivative dx
        dx_gpu = gpuarray.empty((self.model.batch_size, *self.shape), self.model.dtype)
        self.dx = TensorGPU(dx_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        self.states_size = cudnn.cudnnDropoutGetStatesSize(self.model.cudnn_handle)
        self.space_size = cudnn.cudnnDropoutGetReserveSpaceSize(self.y.desc)

        space_gpu = gpuarray.empty((self.space_size.value,), np.byte)
        self.space = TensorGPU(space_gpu, self.model.tensor_format, self.model.cudnn_dtype, TensorGPU.TensorTypeEnum.OTHER)

        states_gpu = gpuarray.empty((self.states_size.value,), np.byte)
        self.states = TensorGPU(states_gpu, self.model.tensor_format, self.model.cudnn_dtype, TensorGPU.TensorTypeEnum.OTHER)

        self.drop_desc = cudnn.cudnnCreateDropoutDescriptor()

        cudnn.cudnnSetDropoutDescriptor(self.drop_desc, self.model.cudnn_handle, self.rate,
                                        self.states.ptr, self.states_size, seed=0)

    def forward(self, x: TensorGPU) -> TensorGPU:
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CUDNN)
        cudnn.cudnnDropoutForward(self.model.cudnn_handle, self.drop_desc,
                                  x.desc, x.ptr,
                                  self.y.desc, self.y.ptr,
                                  self.space.ptr, self.space_size.value)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return self.y

    def backward(self, dy: TensorGPU) -> TensorGPU:
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_CUDNN_DX)
        # Compute dx
        cudnn.cudnnDropoutBackward(self.model.cudnn_handle, self.drop_desc,
                                   dy.desc, dy.ptr,
                                   self.dx.desc, self.dx.ptr,
                                   self.space.ptr, self.space_size.value)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return self.dx
