# https://github.com/storypku/cuda-support-for-bazel/blob/9a9c90c7d73fdafb3fbc8713232405cae4ae66d8/examples/cudnn-samples/multiHeadAttention/multiHeadAttention.cpp
import numpy as np

from pydtnn.libs import libcudnn as cudnn
import pycuda.gpuarray as gpuarray
import pycuda

from pydtnn.backends.gpu.layers.layer import LayerGPU
from pydtnn.backends.gpu.utils.tensor_gpu import TensorGPU
from pydtnn.layers.multi_head_attention import MultiHeadAttention
from pydtnn.utils import initializers
from pydtnn.backends.gpu.utils.memory_allocation import checkConvolutionMemory, getConvolutionWorkspaceSize, getConvolutionWorkspacePtr


class MultiHeadAttentionGPU(MultiHeadAttention[TensorGPU], LayerGPU):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beam = 1  # Number of hypothesis we keep

        self.grad_vars = {"weights": "dw"}

        # The next attributes will be initialized later
        self.y = self.dx = None

    def initialize(self, prev_shape, x):
        super().initialize(prev_shape, x)
        self.query, self.key, self.value, self.mask = x
        self.shape = prev_shape

        self.batch = self.query.shape[0]
        self.seq = self.query.shape[-2]
        self.embedl = self.query.shape[-1]

        # Weights Initializers
        weights_shape = (self.embedl, self.heads*self.d_k)
        biases_shape = (1, self.heads*self.d_k)
        o_weights_shape = (self.heads*self.d_k, self.embedl)
        o_biases_shape = (1, self.embedl)

        self.q_weights_cpu = self.weights_initializer(weights_shape, self.model.dtype)
        self.k_weights_cpu = self.weights_initializer(weights_shape, self.model.dtype)
        self.v_weights_cpu = self.weights_initializer(weights_shape, self.model.dtype)
        self.q_biases_cpu = self.biases_initializer(biases_shape, self.model.dtype)
        self.k_biases_cpu = self.biases_initializer(biases_shape, self.model.dtype)
        self.v_biases_cpu = self.biases_initializer(biases_shape, self.model.dtype)
        self.o_weights_cpu = self.weights_initializer(o_weights_shape, self.model.dtype)
        self.o_biases_cpu = self.biases_initializer(o_biases_shape, self.model.dtype)
        self.nparams = 0
        _weights = [self.q_weights_cpu, self.k_weights_cpu, self.v_weights_cpu, self.o_weights_cpu,
                    self.q_biases_cpu, self.k_biases_cpu, self.v_biases_cpu, self.o_biases_cpu]
        for w in _weights:
            self.nparams += w.size

        # Dropout Descriptor
        self.states_size = cudnn.cudnnDropoutGetStatesSize(self.model.cudnn_handle)
        states_gpu = gpuarray.empty((self.states_size.value,), self.model.dtype)
        self.states = TensorGPU(states_gpu, self.model.tensor_fmt, self.model.cudnn_dtype, TensorGPU.TensorTypeEnum.OTHER)
        self.drop_desc = cudnn.cudnnCreateDropoutDescriptor()
        cudnn.cudnnSetDropoutDescriptor(self.drop_desc, self.model.cudnn_handle, self.dropout_rate,
                                        self.states.ptr, self.states_size.value, seed=0)

        # Attention Descriptor
        self.attn_mode = cudnn.cudnnAttnMode["CUDNN_ATTN_QUERYMAP_ONE_TO_ONE"]
        self.attn_mode += cudnn.cudnnAttnMode["CUDNN_ATTN_ENABLE_PROJ_BIASES"]
        self.add_grad = cudnn.cudnnWgradMode["CUDNN_WGRAD_MODE_SET"]
        self.attn_desc = cudnn.cudnnCreateAttnDescriptor()
        cudnn.cudnnSetAttnDescriptor(self.attn_desc, self.attn_mode, self.heads, 1.0, self.model.cudnn_dtype, self.model.cudnn_dtype, cudnn.cudnnMathType['CUDNN_DEFAULT_MATH'],
                                     None, None, self.embedl, self.embedl, self.embedl, self.d_k, self.d_k, self.d_k, self.embedl,
                                     self.seq, self.seq, self.model.batch_size, self.beam)

        # GPU Memory Allocation
        self.weights_size, self.workspace_size, self.reserve_backward_size = cudnn.cudnnGetMultiHeadAttnBuffers(self.model.cudnn_handle, self.attn_desc)
        checkConvolutionMemory(self.workspace_size)
        weights_size = self.weights_size.value // np.dtype(self.model.dtype).itemsize
        reserve_backward_size = self.reserve_backward_size.value // np.dtype(self.model.dtype).itemsize + 1
        self.weights = gpuarray.empty((weights_size,), self.model.dtype)
        self.weights = TensorGPU(self.weights, self.model.tensor_fmt, self.model.cudnn_dtype, TensorGPU.TensorTypeEnum.OTHER)
        self.dw = gpuarray.empty((weights_size,), self.model.dtype)
        self.dw = TensorGPU(self.dw, self.model.tensor_fmt, self.model.cudnn_dtype, TensorGPU.TensorTypeEnum.OTHER)
        self.reserve_backward = gpuarray.empty((reserve_backward_size,), self.model.dtype)
        self.reserve_backward = TensorGPU(self.reserve_backward, self.model.tensor_fmt, self.model.cudnn_dtype, TensorGPU.TensorTypeEnum.OTHER)

        # Weights to GPU
        # self.copy_weights()
        _weights_types = ["CUDNN_MH_ATTN_Q_WEIGHTS", "CUDNN_MH_ATTN_K_WEIGHTS", "CUDNN_MH_ATTN_V_WEIGHTS", "CUDNN_MH_ATTN_O_WEIGHTS",
                          "CUDNN_MH_ATTN_Q_BIASES", "CUDNN_MH_ATTN_K_BIASES", "CUDNN_MH_ATTN_V_BIASES", "CUDNN_MH_ATTN_O_BIASES"]
        _weights = [self.q_weights_cpu, self.k_weights_cpu, self.v_weights_cpu, self.o_weights_cpu,
                    self.q_biases_cpu, self.k_biases_cpu, self.v_biases_cpu, self.o_biases_cpu]
        pycuda.driver.memcpy_htod(self.weights.ptr.value, np.concatenate([w.flatten() for w in _weights]))

        # Memory Allocation for Outputs
        self.y = gpuarray.empty((self.model.batch_size, self.beam, self.seq, self.embedl), self.model.dtype)
        self.y = TensorGPU(self.y, self.model.tensor_fmt, self.model.cudnn_dtype, TensorGPU.TensorTypeEnum.SEQ)
        self.dquery = gpuarray.empty((self.model.batch_size, self.beam,  self.seq, self.embedl), self.model.dtype)
        self.dquery = TensorGPU(self.dquery, self.model.tensor_fmt, self.model.cudnn_dtype, TensorGPU.TensorTypeEnum.SEQ)
        self.dkey = gpuarray.empty((self.model.batch_size,  self.beam, self.seq, self.embedl), self.model.dtype)
        self.dkey = TensorGPU(self.dkey, self.model.tensor_fmt, self.model.cudnn_dtype, TensorGPU.TensorTypeEnum.SEQ)
        self.dvalue = gpuarray.empty((self.model.batch_size,  self.beam, self.seq, self.embedl), self.model.dtype)
        self.dvalue = TensorGPU(self.dvalue, self.model.tensor_fmt, self.model.cudnn_dtype, TensorGPU.TensorTypeEnum.SEQ)

        self.current_index = -1  # Training
        self.low_window_index = np.full(shape=(self.batch, self.beam, self.seq), fill_value=0, dtype=np.int32)
        self.high_window_index = np.full(shape=(self.batch, self.beam, self.seq), fill_value=self.seq, dtype=np.int32)
        # self.dev_seq_lengths_QO = np.full(shape=(self.batch*self.beam), fill_value=self.seq, dtype=np.int32)
        self.dev_seq_lengths_QO = np.copy(self.y.seq_length_array)
        self.dev_seq_lengths_QO = gpuarray.to_gpu(self.dev_seq_lengths_QO)
        # self.dev_seq_lengths_KV = np.full(shape=(self.batch*self.beam), fill_value=self.seq, dtype=np.int32)
        self.dev_seq_lengths_KV = np.copy(self.dkey.seq_length_array)
        self.dev_seq_lengths_KV = gpuarray.to_gpu(self.dev_seq_lengths_KV)

    def copy_weights(self):
        _weights_types = ["CUDNN_MH_ATTN_Q_WEIGHTS", "CUDNN_MH_ATTN_K_WEIGHTS", "CUDNN_MH_ATTN_V_WEIGHTS", "CUDNN_MH_ATTN_O_WEIGHTS",
                          "CUDNN_MH_ATTN_Q_BIASES", "CUDNN_MH_ATTN_K_BIASES", "CUDNN_MH_ATTN_V_BIASES", "CUDNN_MH_ATTN_O_BIASES"]
        _weights = [self.q_weights_cpu, self.k_weights_cpu, self.v_weights_cpu, self.o_weights_cpu,
                    self.q_biases_cpu, self.k_biases_cpu, self.v_biases_cpu, self.o_biases_cpu]

        pycuda.driver.memcpy_htod(self.weights.ptr.value, np.concatenate([w.flatten() for w in _weights]))
        return

        for i in range(len(_weights)):
            wDesc, dest = cudnn.cudnnGetMultiHeadAttnWeights(self.model.cudnn_handle, self.attn_desc,
                                                             cudnn.cudnnMultiHeadAttnWeightKind[_weights_types[i]], self.weights_size.value, self.weights.ptr)
            print(_weights_types[i], dest)
            # Check wDesc order matches
            if dest is not None:
                pycuda.driver.memcpy_htod(dest[0], _weights[i])

    def forward(self, query, key, value, mask=None, residuals=None):
        if True:  # self.model.mode == Model.Mode.TRAIN:
            self.query, self.key, self.value = query, key, value
            # return self.query
            cudnn.cudnnMultiHeadAttnForward(self.model.cudnn_handle, self.attn_desc,
                                            self.current_index, self.low_window_index, self.high_window_index,
                                            self.dev_seq_lengths_QO.ptr, self.dev_seq_lengths_KV.ptr,
                                            self.dquery.desc, query.ptr,
                                            residuals.ptr,
                                            self.dkey.desc, key.ptr,
                                            self.dvalue.desc, value.ptr,
                                            self.y.desc, self.y.ptr,
                                            self.weights_size.value, self.weights.ptr,
                                            getConvolutionWorkspaceSize(), getConvolutionWorkspacePtr(),
                                            self.reserve_backward_size.value, self.reserve_backward.ptr)
        else:
            cudnn.cudnnMultiHeadAttnForward(self.model.cudnn_handle, self.attn_desc,
                                            self.current_index, self.low_window_index, self.high_window_index,
                                            self.dev_seq_lengths_QO.ptr, self.dev_seq_lengths_KV.ptr,
                                            self.dquery.desc, query.ptr,
                                            residuals.ptr,
                                            self.dkey.desc, key.ptr,
                                            self.dvalue.desc, value.ptr,
                                            self.y.desc, self.y.ptr,
                                            self.weights_size.value, self.weights.ptr,
                                            getConvolutionWorkspaceSize(), getConvolutionWorkspacePtr(),
                                            0, None)
        return self.y

    def backward(self, dy):
        cudnn.cudnnMultiHeadAttnBackwardData(self.model.cudnn_handle, self.attn_desc,
                                             self.low_window_index, self.high_window_index,
                                             self.dev_seq_lengths_QO.ptr, self.dev_seq_lengths_KV.ptr,
                                             self.y.desc, dy.ptr,
                                             self.dquery.desc, self.dquery.ptr, self.query.ptr,
                                             self.dkey.desc, self.dkey.ptr, self.key.ptr,
                                             self.dvalue.desc, self.dvalue.ptr, self.value.ptr,
                                             self.weights_size.value, self.weights.ptr,
                                             getConvolutionWorkspaceSize(), getConvolutionWorkspacePtr(),
                                             self.reserve_backward_size.value, self.reserve_backward.ptr)

        cudnn.cudnnMultiHeadAttnBackwardWeights(self.model.cudnn_handle, self.attn_desc, self.add_grad,
                                                self.dquery.desc, self.query.ptr,
                                                self.dkey.desc, self.key.ptr,
                                                self.dvalue.desc, self.value.ptr,
                                                self.y.desc, dy.ptr,
                                                self.weights_size.value, self.weights.ptr, self.dw.ptr,
                                                getConvolutionWorkspaceSize(), getConvolutionWorkspacePtr(),
                                                self.reserve_backward_size.value, self.reserve_backward.ptr)

        return self.dquery, self.dkey, self.dvalue
