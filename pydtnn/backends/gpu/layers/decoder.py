import numpy as np

from pydtnn.backends.gpu.libs import libcudnn as cudnn

from pydtnn.backends.gpu.layers.abstract.block_layer import AbstractBlockLayerGPU
from pydtnn.layers.multi_head_attention import MultiHeadAttention
from pydtnn.layers.dropout import Dropout
from pydtnn.layers.feed_forward import FeedForward
from pydtnn.layers.layer_normalization import LayerNormalization
from pydtnn.backends.gpu.utils.tensor_gpu import TensorGPU
from pydtnn.layers.decoder import Decoder


class DecoderGPU(AbstractBlockLayerGPU, Decoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multiheadattention = MultiHeadAttention(embedl=self.embedl, d_k=self.d_k, heads=self.heads, dropout_rate=self.dropout_rate)
        # self.dropout_1 = DropoutGPU(rate=self.dropout_rate)
        self.layernormalization_1 = LayerNormalization(axis=(3,))
        self.multiheadattention_enc = MultiHeadAttention(embedl=self.embedl, d_k=self.d_k, heads=self.heads, dropout_rate=self.dropout_rate)
        # self.dropout_enc = DropoutGPU(rate=self.dropout_rate)
        self.layernormalization_enc = LayerNormalization(axis=(3,))
        self.feedforward = FeedForward(shape=(self.embedl,), d_ff=self.d_ff, dropout_rate=self.dropout_rate)
        self.dropout_2 = Dropout(rate=self.dropout_rate)
        self.layernormalization_2 = LayerNormalization(axis=(3,))
        self.paths = [[self.multiheadattention, self.layernormalization_1, self.multiheadattention_enc, self.layernormalization_enc, self.feedforward, self.dropout_2, self.layernormalization_2]]

        # The next attributes will be initialized later
        self.y = self.dx = None

    def initialize(self, prev_shape, x):
        super().initialize(prev_shape, x)
        x_dec, x_enc, mask_dec = x
        x_dec_shape, x_enc_shape, mask_dec_shape = prev_shape
        mha_shape = (x_dec_shape, mask_dec_shape)

        self.shape = x_enc_shape
        self.first_dims = x_dec.ary.shape[:-1]

        # Initialize all sublayers
        for layer in self.children:
            layer.init_backend_from_model(self.model)

        self.multiheadattention.initialize(prev_shape=mha_shape, x=(x_dec, x_dec, x_dec, mask_dec))
        self.layernormalization_1.initialize(prev_shape=self.shape, x=self.multiheadattention.y)
        self.multiheadattention_enc.initialize(prev_shape=mha_shape, x=(x_dec, x_enc, x_enc, mask_dec))
        self.layernormalization_enc.initialize(prev_shape=self.shape, x=self.multiheadattention_enc.y)

        x_aux = self.flatten(self.layernormalization_enc.y)
        self.layernormalization_enc_y_flatten = TensorGPU(x_aux.ary, self.model.tensor_fmt, self.model.cudnn_dtype)

        self.feedforward.initialize(prev_shape=self.layernormalization_enc.shape, x=self.layernormalization_enc_y_flatten)

        x_aux = self.unflatten(self.feedforward.y)
        self.feedforward_y_unflatten = TensorGPU(x_aux.ary, self.model.tensor_fmt, self.model.cudnn_dtype)
        x_aux = self.unflatten(self.feedforward.dx)
        self.feedforward_dx_unflatten = TensorGPU(x_aux.ary, self.model.tensor_fmt, self.model.cudnn_dtype)

        self.dropout_2.initialize(prev_shape=self.feedforward.shape, x=self.feedforward_y_unflatten)
        x_aux = self.flatten(self.dropout_2.dx)
        self.dropout_2_dx_flatten = TensorGPU(x_aux.ary, self.model.tensor_fmt, self.model.cudnn_dtype)

        self.layernormalization_2.initialize(prev_shape=self.dropout_2.shape, x=self.dropout_2.y)

        self.y = self.layernormalization_2.y
        x_aux = self.multiheadattention.dquery
        self.dx = TensorGPU(x_aux.ary, self.model.tensor_fmt, self.model.cudnn_dtype)
        x_aux = self.multiheadattention_enc.dkey
        self.dx_enc = TensorGPU(x_aux.ary, self.model.tensor_fmt, self.model.cudnn_dtype)
        x_aux = self.multiheadattention_enc.dquery
        self.dx_enc_dquery = TensorGPU(x_aux.ary, self.model.tensor_fmt, self.model.cudnn_dtype)

        self.flatten(self.feedforward.y)
        self.flatten(self.feedforward.dx)

        for layer in self.children:
            self.fwd_time += layer.fwd_time
            self.bwd_time += layer.bwd_time
            self.nparams += layer.nparams

    def initialize_block_layer(self):
        pass

    def flatten(self, x):
        last_dim = x.ary.shape[-1]
        return x.reshape((int(np.prod(self.first_dims)), last_dim))

    def unflatten(self, x):
        last_dim = x.ary.shape[-1]
        return x.reshape((*self.first_dims, last_dim))

    def forward(self, x, x_enc, mask=None):
        alpha, beta = 1.0, 1.0
        # Self Attention
        self.multiheadattention.forward(x, x, x, mask, x)
        self.layernormalization_1.forward(self.multiheadattention.y)

        # Self Attention Encoder
        self.multiheadattention_enc.forward(self.layernormalization_1.y, x_enc, x_enc, mask, self.layernormalization_1.y)
        self.layernormalization_enc.forward(self.multiheadattention_enc.y)

        # Feed Forward
        self.feedforward.forward(self.layernormalization_enc_y_flatten)
        self.dropout_2.forward(self.feedforward_y_unflatten)
        # y = dropout_2.y + layernormalization_enc.y
        cudnn.cudnnAddTensor(self.model.cudnn_handle,
                             alpha, self.layernormalization_enc.y.desc, self.layernormalization_enc.y.ptr,
                             beta, self.dropout_2.y.desc, self.dropout_2.y.ptr)
        self.layernormalization_2.forward(self.dropout_2.y)
        return self.y

    def backward(self, dy):

        alpha, beta = 1.0, 1.0
        # Feed Forward
        self.layernormalization_2.backward(dy)
        self.dropout_2.backward(self.layernormalization_2.dx)
        self.feedforward.backward(self.dropout_2_dx_flatten)

        # dx = feedforward.dx + layernormalization_2.dx
        cudnn.cudnnAddTensor(self.model.cudnn_handle,
                             alpha, self.layernormalization_2.dx.desc, self.layernormalization_2.dx.ptr,
                             beta, self.feedforward_dx_unflatten.desc, self.feedforward_dx_unflatten.ptr)

        # Self Attention Encoder
        self.layernormalization_enc.backward(self.feedforward_dx_unflatten)
        self.multiheadattention_enc.backward(self.layernormalization_enc.dx)
        # dx_enc = multihead.dkey + multiheadattention.dvalue
        cudnn.cudnnAddTensor(self.model.cudnn_handle,
                             alpha, self.dx_enc.desc, self.multiheadattention_enc.dvalue.ptr,
                             beta, self.dx_enc.desc, self.dx_enc.ptr)
        # dx = layernorm_enc.dx + multihead_enc.dquery
        cudnn.cudnnAddTensor(self.model.cudnn_handle,
                             alpha, self.dx_enc_dquery.desc, self.layernormalization_enc.dx.ptr,
                             beta, self.dx_enc_dquery.desc, self.multiheadattention_enc.dquery.ptr)

        # Self Attention
        self.layernormalization_1.backward(self.dx_enc_dquery)
        self.multiheadattention.backward(self.layernormalization_1.dx)
        # if self.need_dx:
        # dx = layernorm_1.dx + multihead.dquery + multihead.dkey + multihead.dvalue
        cudnn.cudnnAddTensor(self.model.cudnn_handle,
                             alpha, self.dx.desc, self.layernormalization_1.dx.ptr,
                             beta, self.dx.desc, self.dx.ptr)
        cudnn.cudnnAddTensor(self.model.cudnn_handle,
                             alpha, self.dx.desc, self.multiheadattention.dkey.ptr,
                             beta, self.dx.desc, self.dx.ptr)
        cudnn.cudnnAddTensor(self.model.cudnn_handle,
                             alpha, self.dx.desc, self.multiheadattention.dvalue.ptr,
                             beta, self.dx.desc, self.dx.ptr)
        return self.dx, self.dx_enc
        # else:
        #     return self.dx_enc
