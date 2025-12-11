import numpy as np

from pydtnn.backends.cpu.layers.abstract.block_layer import AbstractBlockLayerCPU
from pydtnn.layers.multi_head_attention import MultiHeadAttention
from pydtnn.layers.dropout import Dropout
from pydtnn.layers.feed_forward import FeedForward
from pydtnn.layers.layer_normalization import LayerNormalization
from pydtnn.layers.encoder import Encoder
from pydtnn.tracers.events import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_OPS_EVENT_enum


class EncoderCPU(Encoder[np.ndarray], AbstractBlockLayerCPU):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multiheadattention = MultiHeadAttention(embedl=self.embedl, d_k=self.d_k, heads=self.heads, dropout_rate=self.dropout_rate)
        # self.dropout_1 = Dropout(rate=self.dropout_rate)
        self.layernormalization_1 = LayerNormalization(axis=(-1,))
        self.feedforward = FeedForward(shape=(self.embedl,), d_ff=self.d_ff, dropout_rate=self.dropout_rate)
        self.dropout_2 = Dropout(rate=self.dropout_rate)
        self.layernormalization_2 = LayerNormalization(axis=(-1,))
        # self.paths = [[self.multiheadattention, self.dropout_1, self.layernormalization_1, self.feedforward,
        #                self.dropout_2, self.layernormalization_2]]
        self.paths = [[self.multiheadattention, self.layernormalization_1, self.feedforward,
                       self.dropout_2, self.layernormalization_2]]

    def initialize(self, prev_shape, x):
        super().initialize(prev_shape, x)
        self.y = x
        if type(x) is tuple:
            x_enc, mask_enc = x
            x_enc_shape, mask_enc_shape = prev_shape
        else:
            x_enc = x
            x_enc_shape = prev_shape
            mask_enc = None
            mask_enc_shape = ()

        self.shape = x_enc_shape
        self.first_dims = x_enc.ary.shape[:-1]

        # Initialize all sublayers
        for layer in self.children:
            layer.init_backend_from_model(self.model)

        self.multiheadattention.initialize(prev_shape=prev_shape, x=(x_enc, x_enc, x_enc, mask_enc))
        # self.dropout_1.initialize(prev_shape=self.multiheadattention.shape, x=self.multiheadattention.y)
        self.layernormalization_1.initialize(prev_shape=self.shape, x=self.multiheadattention.y)

        self.layernormalization_1_y_flatten = None  # self.flatten(self.layernormalization_1.y)).copy()

        self.feedforward.initialize(prev_shape=self.layernormalization_1.shape, x=self.layernormalization_1_y_flatten)

        self.feedforward_y_unflatten = None  # self.unflatten(self.feedforward.y).copy()
        self.feedforward_dx_unflatten = None  # self.unflatten(self.feedforward.dx).copy()

        self.dropout_2.initialize(prev_shape=self.feedforward.shape, x=self.feedforward_y_unflatten)
        self.dropout_2_dx_flatten = None  # self.flatten(self.dropout_2.dx).copy()

        self.layernormalization_2.initialize(prev_shape=self.dropout_2.shape, x=self.dropout_2.y)

        self.y = self.layernormalization_2.y
        # x_aux = self.multiheadattention.dquery
        self.dx = None  # x_aux.copy()

        # self.flatten(self.feedforward.y)  # FeedForward uses shape while LayerNormalization and Dropout dont
        # self.flatten(self.feedforward.dx)  # FeedForward uses shape while LayerNormalization and Dropout dont

        for layer in self.children:
            self.fwd_time += layer.fwd_time
            self.bwd_time += layer.bwd_time
            self.nparams += layer.nparams

    def initialize_block_layer(self):
        pass

    def forward(self, x, mask=None):
        # Self Attention
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_MHA)
        x_1 = self.multiheadattention.forward(x, x, x, mask)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        x_1 = self.dropout_1.forward(x_1)
        x_1 = x_1 + x
        x_1 = self.layernormalization_1.forward(x_1)
        # Feed Forward
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_FEEDFORWARD)
        x_2 = self.feedforward.forward(x_1)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        x_2 = self.dropout_2.forward(x_2)
        x_2 = x_1 + x_2
        self.y = self.layernormalization_2.forward(x_2)
        return self.y

    def backward(self, dy):
        # Feed Forward
        dx_1 = self.layernormalization_2.backward(dy)
        dx_2 = self.dropout_2.backward(dx_1)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_FEEDFORWARD)
        dx_2 = self.feedforward.backward(dx_2)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        dx_2 = dx_1 + dx_2
        # Self Attention
        dx_2 = self.layernormalization_1.backward(dx_2)
        dx_3 = self.dropout_1.backward(dx_2)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_MHA)
        dx_3 = self.multiheadattention.backward(dx_3)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        dx_3, dx_4, dx_5 = dx_3
        self.dx = dx_2 + dx_3 + dx_4 + dx_5
        return self.dx
