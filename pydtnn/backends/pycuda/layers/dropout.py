"""PyCUDA implementation of the Dropout layer for the PyDTNN framework."""

import logging
from typing import Any

import numpy as np

from pydtnn.backends.pycuda.layers.abstract.layer import LayerPycuda
from pydtnn.layers.dropout import Dropout
from pydtnn.libs import cudnn as cudnn
from pydtnn.tracers.abstract.events import (PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT,
                                            PYDTNN_OPS_EVENTS, OpsEventEnum)
from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.tensor_array import FormatType, MemoryType, TensorArray

__all__ = ("DropoutPycuda",)

logger = logging.getLogger(__name__)


class DropoutPycuda(Dropout[TensorArray], LayerPycuda):
    """PyCUDA-accelerated Dropout layer using cuDNN."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the DropoutPycuda layer."""
        super().__init__(*args, **kwargs)

        # The following values will be initalized later:
        self.states_size: int = None  # pyright: ignore[reportAttributeAccessIssue]
        self.space_size: int = None  # pyright: ignore[reportAttributeAccessIssue]
        self.space: TensorArray = None  # pyright: ignore[reportAttributeAccessIssue]
        self.states: TensorArray = None  # pyright: ignore[reportAttributeAccessIssue]
        self.drop_desc: int = None  # pyright: ignore[reportAttributeAccessIssue]

    def _model_init(self, prev_shape: ArrayShape, x: TensorArray) -> None:
        """Initializes cuDNN descriptors and memory buffers for dropout operations."""
        super()._model_init(prev_shape, x)

        # Activations y
        self.y = TensorArray.zeros(
            (self.model.batch_size, *self.shape),
            self.model.dtype,
            tensor_format=self.model.tensor_format,
            mem_type=MemoryType.GLOBAL if self.model.use_gpudirect else MemoryType.DEVICE,
            fmt_type=FormatType.TENSOR
        )
        self.memory_used += self.y.nbytes

        # Derivative dx
        self.dx = TensorArray.zeros(
            (self.model.batch_size, *self.shape),
            self.model.dtype,
            tensor_format=self.model.tensor_format,
            mem_type=MemoryType.GLOBAL if self.model.use_gpudirect else MemoryType.DEVICE,
            fmt_type=FormatType.TENSOR
        )
        self.memory_used += self.dx.nbytes

        space_size = cudnn.cudnnDropoutGetReserveSpaceSize(self.y.desc)
        self.space = TensorArray.zeros(
            (space_size,),
            np.byte,
            host_shape=True,
            mem_type=MemoryType.DEVICE,
            fmt_type=FormatType.NONE
        )
        self.memory_used += self.space.nbytes

        states_size = cudnn.cudnnDropoutGetStatesSize(self.model.cudnn_handle)
        self.states = TensorArray.zeros(
            (states_size,),
            np.byte,
            host_shape=True,
            mem_type=MemoryType.DEVICE,
            fmt_type=FormatType.NONE
        )
        self.memory_used += self.states.nbytes

        self.drop_desc = cudnn.cudnnCreateDropoutDescriptor()
        cudnn.cudnnSetDropoutDescriptor(
            self.drop_desc,
            self.model.cudnn_handle,
            self.rate,
            self.states.ptr,
            self.states.size,
            seed=self.model.random_seed,
        )

    def forward(self, x: TensorArray) -> TensorArray:
        """Performs the forward pass of the dropout layer."""
        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.FORWARD_CUDNN
        )
        cudnn.cudnnDropoutForward(
            self.model.cudnn_handle,
            self.drop_desc,
            x.desc,
            x.ptr,
            self.y.desc,
            self.y.ptr,
            self.space.ptr,
            self.space.size,
        )
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return self.y

    def backward(self, dy: TensorArray) -> TensorArray:
        """Performs the backward pass of the dropout layer."""
        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.BACKWARD_CUDNN_DX
        )
        # Compute dx
        cudnn.cudnnDropoutBackward(
            self.model.cudnn_handle,
            self.drop_desc,
            dy.desc,
            dy.ptr,
            self.dx.desc,
            self.dx.ptr,
            self.space.ptr,
            self.space.size,
        )
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return self.dx
