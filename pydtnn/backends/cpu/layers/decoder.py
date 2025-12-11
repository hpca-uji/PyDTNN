import numpy as np

from pydtnn.backends.cpu.layers.abstract.block_layer import AbstractBlockLayerCPU
from pydtnn.layers.multi_head_attention import MultiHeadAttention
from pydtnn.layers.dropout import Dropout
from pydtnn.layers.feed_forward import FeedForward
from pydtnn.layers.layer_normalization import LayerNormalization
from pydtnn.layers.decoder import Decoder

from pydtnn.tracers.events import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_OPS_EVENT_enum


class DecoderCPU(Decoder[np.ndarray], AbstractBlockLayerCPU):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multiheadattention = MultiHeadAttention(embedl=self.embedl, d_k=self.d_k, heads=self.heads, dropout_rate=self.dropout_rate)
        # self.dropout_1 = Dropout(rate=self.dropout_rate)
        self.layernormalization_1 = LayerNormalization(axis=(1, 2))
        self.multiheadattention_enc = MultiHeadAttention(embedl=self.embedl, d_k=self.d_k, heads=self.heads, dropout_rate=self.dropout_rate)
        # self.dropout_enc = Dropout(rate=self.dropout_rate)
        self.layernormalization_enc = LayerNormalization(axis=(1, 2))
        self.feedforward = FeedForward(shape=(self.embedl,), d_ff=self.d_ff, dropout_rate=self.dropout_rate)
        self.dropout_2 = Dropout(rate=self.dropout_rate)
        self.layernormalization_2 = LayerNormalization(axis=(1, 2))
        # self.paths = [[self.multiheadattention, self.dropout_1, self.layernormalization_1, self.multiheadattention_enc,
        #                self.dropout_enc, self.layernormalization_enc, self.feedforward, self.dropout_2, self.layernormalization_2]]
        self.paths = [[self.multiheadattention, self.layernormalization_1, self.multiheadattention_enc,
                       self.layernormalization_enc, self.feedforward, self.dropout_2, self.layernormalization_2]]

    def initialize(self, prev_shape, x):
        super().initialize(prev_shape, x)
        x_dec, x_enc, mask_dec = x
        x_dec_shape, x_enc_shape, mask_dec_shape = prev_shape
        mha_shape = (x_dec_shape, mask_dec_shape)

        # Initialize all sublayers
        for layer in self.children:
            layer.init_backend_from_model(self.model)

        self.multiheadattention.initialize(prev_shape=mha_shape, x=(x_dec, x_dec, x_dec, mask_dec))
        self.layernormalization_1.initialize(prev_shape=self.shape, x=self.multiheadattention.y)
        self.multiheadattention_enc.initialize(prev_shape=mha_shape, x=(x_dec, x_enc, x_enc, mask_dec))
        # self.dropout_enc.initialize(prev_shape=prev_shape, x=self.multiheadattention_enc.y)
        self.layernormalization_enc.initialize(prev_shape=self.shape, x=self.multiheadattention_enc.y)

        # x_aux = self.flatten(self.layernormalization_enc.y)
        self.layernormalization_enc_y_flatten = None  # x_aux.copy()

        self.feedforward.initialize(prev_shape=self.layernormalization_enc.shape, x=self.layernormalization_enc_y_flatten)
        # x_aux = self.unflatten(self.feedforward.dx)
        self.feedforward_dx_unflatten = None  # x_aux.copy()

        self.dropout_2.initialize(prev_shape=self.feedforward.shape, x=self.feedforward_y_unflatten)

        self.layernormalization_2.initialize(prev_shape=prev_shape, x=self.dropout_2.y)

        self.y = self.layernormalization_2.y

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
        # Self Attention
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_MHA)
        x_1 = self.multiheadattention.forward(x, x, x, mask)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        # x_1 = self.dropout_1.forward(x_1)
        x_1 += x
        x_1 = self.layernormalization_1.forward(x_1)
        # Encoder-Decoder Attention
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_MHA)
        x_2 = self.multiheadattention_enc.forward(x_1, x_enc, x_enc, mask)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        # x_2 = self.dropout_enc.forward(x_2)
        x_2 += x_1
        x_2 = self.layernormalization_enc.forward(x_2)
        # Feed Forward
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_FEEDFORWARD)
        x_3 = self.feedforward.forward(x_2)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        x_3 = self.dropout_2.forward(x_3)
        x_3 += x_2
        x_3 = self.layernormalization_2.forward(x_3)
        return x_3

    def backward(self, dy):
        # Feed Forward
        dx_1 = self.layernormalization_2.backward(dy)
        dx_2 = self.dropout_2.backward(dx_1)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_FEEDFORWARD)
        dx_2 = self.feedforward.backward(dx_2)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        dx_2 = dx_1 + dx_2
        # Encoder-Decoder Attention
        dx_2 = self.layernormalization_enc.backward(dx_2)
        # dx_3 = self.dropout_enc.backward(dx_2)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_MHA)
        dx_3, dx_4, dx_5 = self.multiheadattention_enc.backward(dx_3)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        dx_enc = dx_4 + dx_5
        dx_2 = dx_2 + dx_3
        # Self Attention
        dx_2 = self.layernormalization_1.backward(dx_2)
        # dx_3 = self.dropout_1.backward(dx_2)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_MHA)
        dx_3 = self.multiheadattention.backward(dx_3)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        # if self.need_dx:
        dx_3, dx_4, dx_5 = dx_3
        dx = dx_2 + dx_3 + dx_4 + dx_5
        return dx, dx_enc
        # else:
        #     return dx_enc
