import numpy as np

from pydtnn.backends.cpu.layers.abstract.block_layer import AbstractBlockLayerCPU
from pydtnn.layers.fc import FC
from pydtnn.layers.dropout import Dropout
from pydtnn.layers.multiplication import Multiplication
from pydtnn.layers.scalar import Scalar
from pydtnn.activations.softmax import Softmax

from pydtnn.layers.multi_head_attention import MultiHeadAttention
from pydtnn.model import Model
from pydtnn.tracers.events import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_OPS_EVENT_enum


class MultiHeadAttentionCPU(MultiHeadAttention[np.ndarray], AbstractBlockLayerCPU):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.FC_q = FC(shape=(self.heads*self.d_k,))  # Dim: embedl x heads*d_k
        self.FC_k = FC(shape=(self.heads*self.d_k,))  # Dim: embedl x heads*d_k
        self.FC_v = FC(shape=(self.heads*self.d_k,))  # Dim: embedl x heads*d_k
        self.FC_o = FC(shape=(self.embedl,))   # Dim: heads*d_k x embedl
        self.mult_qkt = Multiplication()
        self.scalar_dk = Scalar(1.0/np.sqrt(self.d_k))
        self.softmax = Softmax()
        self.dropout = Dropout(rate=self.dropout_rate)
        self.mult_smv = Multiplication()
        self.mult_o = Multiplication()
        self.paths = [[self.FC_q, self.FC_k, self.FC_v, self.mult_qkt, self.scalar_dk, self.FC_o,
                       self.softmax, self.dropout, self.mult_smv, self.mult_o]]

        # The next attributes will be initialized later
        self.mask = None

    def initialize(self, prev_shape, x):
        super().initialize(prev_shape, x)
        self.shape = prev_shape
        self.embedl = prev_shape[-1]
        seq = prev_shape[-2]

        # Initialize all sublayers
        for layer in self.children:
            layer.init_backend_from_model(self.model)

        self.FC_q.initialize(prev_shape=(self.embedl,), x=x)
        self.FC_k.initialize(prev_shape=(self.embedl,), x=self.FC_q.y)
        self.FC_v.initialize(prev_shape=(self.embedl,), x=self.FC_k.y)
        self.mult_qkt.initialize(prev_shape=(1,), x=self.FC_v.y)
        self.scalar_dk.initialize(prev_shape=(1,), x=self.mult_qkt.y)
        self.FC_o.initialize(prev_shape=(self.heads*self.d_k,), x=self.scalar_dk.y)
        self.softmax.initialize(prev_shape=(self.heads, seq, seq,), x=self.FC_o.y)
        self.dropout.initialize(prev_shape=(self.heads, seq, seq,), x=self.softmax.y)
        self.mult_smv.initialize(prev_shape=(1,), x=self.dropout.y)
        self.mult_o.initialize(prev_shape=(1,), x=self.mult_smv.y)

        for layer in self.children:
            if layer.fwd_time is not None:
                self.fwd_time += layer.fwd_time
            if layer.bwd_time is not None:
                self.bwd_time += layer.bwd_time
            if layer.nparams is not None:
                self.nparams += layer.nparams

    def initialize_block_layer(self):
        pass

    def transformation_addheads(self, x):
        return x.reshape((x.shape[:-1] + (self.heads, self.d_k))).swapaxes(-3, -2)

    def transformation_removeheads(self, x):
        return x.swapaxes(-3, -2).reshape((x.shape[:-3] + (x.shape[-2], self.heads*self.d_k)))

    def mask_apply(self, x, mask):
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

    def transpose(self, x):
        return x.swapaxes(-2, -1)

    def forward(self, query, key, value, mask=None):
        if self.model.mode == Model.Mode.TRAIN:
            self.mask = mask

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_MHA_FC_QKV)
        query = self.transformation_addheads(self.FC_q.forward(query))
        key = self.transformation_addheads(self.FC_k.forward(key))
        value = self.transformation_addheads(self.FC_v.forward(value))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_MHA_MATMUL_QK)
        score = self.mult_qkt.forward(query, self.transpose(key))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_MHA_SCALARDK)
        score = self.scalar_dk.forward(score)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        if self.mask is not None:
            score = self.mask_apply(score, mask)
        score = self.softmax.forward(score)
        score = self.dropout.forward(score)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_MHA_MATMUL_SMV)
        score = self.mult_smv.forward(score, value)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_MHA_FC_O)
        score = self.transformation_removeheads(score)
        score = self.FC_o.forward(score)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        return score

    def backward(self, dy):
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_MHA_FC_O)
        dx = self.FC_o.backward(dy)
        dx = self.transformation_addheads(dx)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_MHA_MATMUL_SMV)
        dx, d_value = self.mult_smv.backward(dx)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        dx = self.dropout.backward(dx)
        dx = self.softmax.backward(dx)
        if self.mask is not None:
            dx = self.mask_apply(dx, self.mask)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_MHA_SCALARDK)
        dx = self.scalar_dk.backward(dx)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_MHA_MATMUL_QK)
        d_query, d_key = self.mult_qkt.backward(dx)
        d_key = self.transpose(d_key)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_MHA_FC_QKV)
        d_query = self.FC_q.backward(self.transformation_removeheads(d_query))
        d_key = self.FC_k.backward(self.transformation_removeheads(d_key))
        d_value = self.FC_v.backward(self.transformation_removeheads(d_value))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        return d_query, d_key, d_value
