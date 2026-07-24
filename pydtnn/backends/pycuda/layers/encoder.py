"""PyCUDA implementation of the Transformer Encoder layer."""

import logging
from typing import Any

import numpy as np

from pydtnn.backends.pycuda.layers.abstract.block_layer import AbstractBlockLayerPycuda
from pydtnn.backends.pycuda.libs import libcudnn as cudnn
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.layers.dropout import Dropout
from pydtnn.layers.encoder import Encoder
from pydtnn.layers.feed_forward import FeedForward
from pydtnn.layers.layer_normalization import LayerNormalization
from pydtnn.layers.multi_head_attention import MultiHeadAttention
from pydtnn.utils.constants import ArrayShape

__all__ = ("EncoderPycuda",)


logger = logging.getLogger(__name__)


class EncoderPycuda(Encoder[TensorArray], AbstractBlockLayerPycuda):
    """PyCUDA-accelerated Transformer Encoder layer."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the EncoderPycuda layer with sub-layers and placeholders."""
        super().__init__(*args, **kwargs)
        self.multiheadattention = MultiHeadAttention(
            embedl=self.embedl, d_k=self.d_k, heads=self.heads, dropout_rate=self.dropout_rate
        )
        # self.dropout_1 = Dropout(rate=self.dropout_rate)
        self.layernormalization_1 = LayerNormalization(axis=(3,))
        self.feedforward = FeedForward(
            shape=(self.embedl,), d_ff=self.d_ff, dropout_rate=self.dropout_rate
        )
        self.dropout_2 = Dropout(rate=self.dropout_rate)
        self.layernormalization_2 = LayerNormalization(axis=(3,))
        self.paths = [
            [
                self.multiheadattention,
                self.layernormalization_1,
                self.feedforward,
                self.dropout_2,
                self.layernormalization_2,
            ]
        ]

        # The next attributes will be initialized later
        self.y: TensorArray = None  # pyright: ignore[reportAttributeAccessIssue]
        self.dx: TensorArray = None  # pyright: ignore[reportAttributeAccessIssue]

    def _model_init(self, prev_shape: ArrayShape, x: TensorArray) -> None:
        """Initializes the backend model and sub-layers with input shapes."""
        super()._model_init(prev_shape, x)
        self.y = x
        if type(x) is tuple:
            x_enc, mask_enc = x
            x_enc_shape, mask_enc_shape = prev_shape  # noqa: F841
        else:
            x_enc = x
            x_enc_shape = prev_shape
            mask_enc = None
            mask_enc_shape = ()  # noqa: F841

        self.shape = x_enc_shape  # pyright: ignore[reportAttributeAccessIssue]
        self.first_dims = x_enc.shape[:-1]

        # Initialize all sublayers
        for layer in self.children:
            layer._init_backend_with_model(self.model)

        self.multiheadattention._model_init(
            prev_shape=prev_shape, x=(x_enc, x_enc, x_enc, mask_enc)
        )
        # self.dropout_1.initialize(prev_shape=self.multiheadattention.shape, x=self.multiheadattention.y)
        self.layernormalization_1._model_init(prev_shape=self.shape, x=self.multiheadattention.y)

        self.layernormalization_1_y_flatten = TensorArray(
            self.flatten(self.layernormalization_1.y).ary,
            self.model.tensor_format,
            self.model.cudnn_dtype,
        )

        self.feedforward._model_init(
            prev_shape=self.layernormalization_1.shape, x=self.layernormalization_1_y_flatten
        )

        self.feedforward_y_unflatten = TensorArray(
            self.unflatten(self.feedforward.y).ary, self.model.tensor_format, self.model.cudnn_dtype
        )
        self.feedforward_dx_unflatten = TensorArray(
            self.unflatten(self.feedforward.dx).ary,
            self.model.tensor_format,
            self.model.cudnn_dtype,
        )

        self.dropout_2._model_init(
            prev_shape=self.feedforward.shape, x=self.feedforward_y_unflatten
        )
        self.dropout_2_dx_flatten = TensorArray(
            self.flatten(self.dropout_2.dx).ary, self.model.tensor_format, self.model.cudnn_dtype
        )

        self.layernormalization_2._model_init(prev_shape=self.dropout_2.shape, x=self.dropout_2.y)

        self.y = self.layernormalization_2.y
        x_aux = self.multiheadattention.dquery
        self.dx = TensorArray(x_aux.ary, self.model.tensor_format, self.model.cudnn_dtype)

        self.flatten(
            self.feedforward.y
        )  # FeedForward uses shape while LayerNormalization and Dropout dont
        self.flatten(
            self.feedforward.dx
        )  # FeedForward uses shape while LayerNormalization and Dropout dont

        for layer in self.children:
            self.fwd_time += layer.fwd_time
            self.bwd_time += layer.bwd_time
            self.nparams += layer.nparams

    def initialize_block_layer(self) -> None:
        """Placeholder for block layer initialization."""
        pass

    def flatten(self, x: TensorArray) -> TensorArray:
        """Flattens the input tensor to 2D."""
        last_dim = x.shape[-1]
        return x.reshape((int(np.prod(self.first_dims)), last_dim))

    def unflatten(self, x: TensorArray) -> TensorArray:
        """Restores the input tensor to its original dimensions."""
        last_dim = x.shape[-1]
        return x.reshape((*self.first_dims, last_dim))

    def forward(self, x: TensorArray, mask: TensorArray | None = None) -> TensorArray:
        """Performs the forward pass through the encoder block."""
        # self.model.test("Forward")
        alpha, beta = 1.0, 1.0
        # Self Attention
        self.multiheadattention.forward(x, x, x, mask, x)  # pyright: ignore[reportCallIssue]
        self.layernormalization_1.forward(self.multiheadattention.y)
        # self.layernormalization_1.forward(x)

        # Feed Forward
        self.feedforward.forward(self.layernormalization_1_y_flatten)
        self.dropout_2.forward(self.feedforward_y_unflatten)
        # y = dropout_2.y + layernormalization_1.y
        cudnn.cudnnAddTensor(
            self.model.cudnn_handle,
            alpha,
            self.feedforward_y_unflatten.desc,
            self.layernormalization_1.y.ptr,
            beta,
            self.feedforward_y_unflatten.desc,
            self.feedforward_y_unflatten.ptr_voidp,
        )

        self.layernormalization_2.forward(self.feedforward_y_unflatten)
        return self.y

    def backward(self, dy: TensorArray) -> TensorArray:
        """Performs the backward pass through the encoder block."""
        # self.model.test("Backward")
        alpha, beta = 1.0, 1.0
        # Feed Forward
        self.layernormalization_2.backward(dy)
        self.dropout_2.backward(self.layernormalization_2.dx)

        self.feedforward.backward(self.dropout_2_dx_flatten)

        # dx = feedforward.dx + layernormalization_2.dx
        cudnn.cudnnAddTensor(
            self.model.cudnn_handle,
            alpha,
            self.feedforward_dx_unflatten.desc,
            self.layernormalization_2.dx.ptr,
            beta,
            self.feedforward_dx_unflatten.desc,
            self.feedforward_dx_unflatten.ptr_voidp,
        )

        # Self Attention
        self.layernormalization_1.backward(self.feedforward_dx_unflatten)
        # return self.layernormalization_1.dx
        self.multiheadattention.backward(self.layernormalization_1.dx)
        # if self.need_dx:
        # dx = layernorm_1.dx + multihead.dquery + multihead.dkey + multihead.dvalue
        cudnn.cudnnAddTensor(
            self.model.cudnn_handle,
            alpha,
            self.dx.desc,
            self.layernormalization_1.dx.ptr,
            beta,
            self.dx.desc,
            self.dx.ptr_voidp,
        )
        cudnn.cudnnAddTensor(
            self.model.cudnn_handle,
            alpha,
            self.dx.desc,
            self.multiheadattention.dkey.ptr,
            beta,
            self.dx.desc,
            self.dx.ptr_voidp,
        )
        cudnn.cudnnAddTensor(
            self.model.cudnn_handle,
            alpha,
            self.dx.desc,
            self.multiheadattention.dvalue.ptr,
            beta,
            self.dx.desc,
            self.dx.ptr_voidp,
        )
        return self.dx
