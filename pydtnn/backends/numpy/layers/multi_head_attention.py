"""Multi-head attention layer implementation for the NumPy backend."""

import logging
from typing import TYPE_CHECKING, Any

from pydtnn.activations.softmax import Softmax
from pydtnn.backends.numpy.layers.abstract.block_layer import AbstractBlockLayerNumpy
from pydtnn.layers.dropout import Dropout
from pydtnn.layers.fc import FC
from pydtnn.layers.multi_head_attention import MultiHeadAttention
from pydtnn.layers.multiplication import Multiplication
from pydtnn.layers.scalar import Scalar
from pydtnn.libs import numpy as np
from pydtnn.model.base import ModelMode
from pydtnn.tracers.events import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, OpsEventEnum
from pydtnn.utils.constants import ArrayShape

__all__ = ("MultiHeadAttentionNumpy",)


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)


class MultiHeadAttentionNumpy(MultiHeadAttention[np.ndarray], AbstractBlockLayerNumpy):
    """NumPy implementation of the Multi-Head Attention mechanism."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the MultiHeadAttentionNumpy layer with required sublayers."""
        super().__init__(*args, **kwargs)
        self.FC_q = FC[np.ndarray](shape=(self.heads * self.d_k,))  # Dim: embedl x heads*d_k
        self.FC_k = FC[np.ndarray](shape=(self.heads * self.d_k,))  # Dim: embedl x heads*d_k
        self.FC_v = FC[np.ndarray](shape=(self.heads * self.d_k,))  # Dim: embedl x heads*d_k
        self.FC_o = FC[np.ndarray](shape=(self.embedl,))  # Dim: heads*d_k x embedl
        self.mult_qkt = Multiplication[np.ndarray]()
        self.scalar_dk = Scalar[np.ndarray](1.0 / np.sqrt(self.d_k))
        self.softmax = Softmax[np.ndarray]()
        self.dropout = Dropout[np.ndarray](rate=self.dropout_rate)
        self.mult_smv = Multiplication[np.ndarray]()
        self.mult_o = Multiplication[np.ndarray]()
        self.paths = [
            [
                self.FC_q,
                self.FC_k,
                self.FC_v,
                self.mult_qkt,
                self.scalar_dk,
                self.FC_o,
                self.softmax,
                self.dropout,
                self.mult_smv,
                self.mult_o,
            ]
        ]

        # The next attributes will be initialized later
        self.mask: np.ndarray = None  # pyright: ignore[reportAttributeAccessIssue]

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray) -> None:
        """Initializes the model structure and sublayers for the NumPy backend."""
        super()._model_init(prev_shape, x)
        self.shape = prev_shape
        if type(prev_shape[0]) is tuple:
            enc_shape = prev_shape[0]
        else:
            enc_shape = prev_shape
        self.embedl = enc_shape[-1]
        seq = enc_shape[-2]

        # Initialize all sublayers
        for layer in self.children:
            layer._init_backend_with_model(self.model)

        self.FC_q._model_init(prev_shape=(self.embedl,), x=x)
        self.FC_k._model_init(prev_shape=(self.embedl,), x=self.FC_q.y)
        self.FC_v._model_init(prev_shape=(self.embedl,), x=self.FC_k.y)
        self.mult_qkt._model_init(prev_shape=(1,), x=self.FC_v.y)
        self.scalar_dk._model_init(prev_shape=(1,), x=self.mult_qkt.y)
        self.FC_o._model_init(prev_shape=(self.heads * self.d_k,), x=self.scalar_dk.y)
        self.softmax._model_init(
            prev_shape=(
                self.heads,
                seq,
                seq,
            ),
            x=self.FC_o.y,
        )
        self.dropout._model_init(
            prev_shape=(
                self.heads,
                seq,
                seq,
            ),
            x=self.softmax.y,
        )
        self.mult_smv._model_init(prev_shape=(1,), x=self.dropout.y)
        self.mult_o._model_init(prev_shape=(1,), x=self.mult_smv.y)

        for layer in self.children:
            if layer.fwd_time is not None:
                self.fwd_time += layer.fwd_time
            if layer.bwd_time is not None:
                self.bwd_time += layer.bwd_time
            if layer.nparams is not None:
                self.nparams += layer.nparams

    def initialize_block_layer(self) -> None:
        """Placeholder for block layer initialization."""
        pass

    def transformation_addheads(self, x: np.ndarray) -> np.ndarray:
        """Reshapes input to separate attention heads."""
        return x.reshape((x.shape[:-1] + (self.heads, self.d_k))).swapaxes(-3, -2)

    def transformation_removeheads(self, x: np.ndarray) -> np.ndarray:
        """Reshapes input to merge attention heads."""
        return x.swapaxes(-3, -2).reshape((x.shape[:-3] + (x.shape[-2], self.heads * self.d_k)))

    def mask_apply(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Applies an attention mask to the score tensor."""
        if len(mask.shape) == 2:
            seq, seq2 = mask.shape
        else:
            _, seq, seq2 = mask.shape
        if seq == 1:
            for j in range(self.heads):
                for k in range(seq2):
                    x[:, j, k] = x[:, j, k] * mask[:, 0]
        else:
            for j in range(self.heads):
                x[:, j] = x[:, j] * mask[:]
        return x

    def transpose(self, x: np.ndarray) -> np.ndarray:
        """Transposes the last two dimensions of the input."""
        return x.swapaxes(-2, -1)

    def forward(
        self, query: np.ndarray, key: np.ndarray, value: np.ndarray, mask: np.ndarray | None = None
    ) -> np.ndarray:
        """Performs the forward pass of the multi-head attention mechanism."""
        if self.model.mode == ModelMode.TRAIN:
            # TODO: Check this. (in this case, mask is not None) (I hope)
            self.mask = mask  # pyright: ignore[reportAttributeAccessIssue]

        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.FORWARD_MHA_FC_QKV
        )
        query = self.transformation_addheads(self.FC_q.forward(query))
        key = self.transformation_addheads(self.FC_k.forward(key))
        value = self.transformation_addheads(self.FC_v.forward(value))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT,
            self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.FORWARD_MHA_MATMUL_QK,
        )

        score = self.mult_qkt.forward(query, self.transpose(key))  # pyright: ignore[reportCallIssue]
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT,
            self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.FORWARD_MHA_SCALARDK,
        )
        score = self.scalar_dk.forward(score)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        if self.mask is not None:
            score = self.mask_apply(score, self.mask)
        score = self.softmax.forward(score)
        score = self.dropout.forward(score)

        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT,
            self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.FORWARD_MHA_MATMUL_SMV,
        )

        score = self.mult_smv.forward(score, value)  # pyright: ignore[reportCallIssue]
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.FORWARD_MHA_FC_O
        )
        score = self.transformation_removeheads(score)
        score = self.FC_o.forward(score)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        return score

    def backward(self, dy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Performs the backward pass of the multi-head attention mechanism."""
        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.BACKWARD_MHA_FC_O
        )
        dx = self.FC_o.backward(dy)
        dx = self.transformation_addheads(dx)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT,
            self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.BACKWARD_MHA_MATMUL_SMV,
        )
        dx, d_value = self.mult_smv.backward(dx)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        dx = self.dropout.backward(dx)
        dx = self.softmax.backward(dx)
        if self.mask is not None:
            dx = self.mask_apply(dx, self.mask)

        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT,
            self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.BACKWARD_MHA_SCALARDK,
        )
        dx = self.scalar_dk.backward(dx)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT,
            self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.BACKWARD_MHA_MATMUL_QK,
        )
        d_query, d_key = self.mult_qkt.backward(dx)
        d_key = self.transpose(d_key)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT,
            self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.BACKWARD_MHA_FC_QKV,
        )
        d_query = self.FC_q.backward(self.transformation_removeheads(d_query))
        d_key = self.FC_k.backward(self.transformation_removeheads(d_key))
        d_value = self.FC_v.backward(self.transformation_removeheads(d_value))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        return d_query, d_key, d_value
