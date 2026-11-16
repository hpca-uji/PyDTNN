"""PyDTNN PyCUDA backend implementation for Multi-Head Attention layers."""

# https://github.com/storypku/cuda-support-for-bazel/blob/9a9c90c7d73fdafb3fbc8713232405cae4ae66d8/examples/cudnn-samples/multiHeadAttention/multiHeadAttention.cpp

import ctypes
import logging
from typing import Any

import numpy as np

from pycuda import driver as drv
from pydtnn.backends.pycuda.layers.abstract.layer import LayerPycuda
from pydtnn.layers.multi_head_attention import MultiHeadAttention
from pydtnn.libs import cudnn as cudnn
from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.tensor_array import DescriptorType, MemoryType, TensorArray

__all__ = ("MultiHeadAttentionPycuda",)

logger = logging.getLogger(__name__)


class MultiHeadAttentionPycuda(MultiHeadAttention[TensorArray], LayerPycuda):
    """PyCUDA implementation of the Multi-Head Attention layer using cuDNN."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the MultiHeadAttentionPycuda layer."""
        super().__init__(*args, **kwargs)
        self.beam = 1  # Number of hypothesis we keep

        self.grad_vars = {"weights": "dw"}

        # The next attributes will be initialized later
        self.y: TensorArray = None  # pyright: ignore[reportAttributeAccessIssue]
        self.dx: TensorArray = None  # pyright: ignore[reportAttributeAccessIssue]

    def _model_init(self, prev_shape: ArrayShape, x: TensorArray) -> None:
        """Initializes model parameters, cuDNN descriptors, and GPU memory buffers."""
        super()._model_init(prev_shape, x)
        self.query, self.key, self.value, self.mask = x
        self.shape = prev_shape

        self.batch = self.query.shape[0]
        self.seq = self.query.shape[-2]
        self.embedl = self.query.shape[-1]

        # Weights Initializers
        weights_shape = (self.embedl, self.heads * self.d_k)
        biases_shape = (1, self.heads * self.d_k)
        o_weights_shape = (self.heads * self.d_k, self.embedl)
        o_biases_shape = (1, self.embedl)

        self.q_weights_cpu = self.weights_initializer(
            weights_shape, self.model.param_dtype, self.model.random
        )
        self.k_weights_cpu = self.weights_initializer(
            weights_shape, self.model.param_dtype, self.model.random
        )
        self.v_weights_cpu = self.weights_initializer(
            weights_shape, self.model.param_dtype, self.model.random
        )
        self.q_biases_cpu = self.biases_initializer(
            biases_shape, self.model.param_dtype, self.model.random
        )
        self.k_biases_cpu = self.biases_initializer(
            biases_shape, self.model.param_dtype, self.model.random
        )
        self.v_biases_cpu = self.biases_initializer(
            biases_shape, self.model.param_dtype, self.model.random
        )
        self.o_weights_cpu = self.weights_initializer(
            o_weights_shape, self.model.param_dtype, self.model.random
        )
        self.o_biases_cpu = self.biases_initializer(
            o_biases_shape, self.model.param_dtype, self.model.random
        )
        self.nparams = 0
        _weights = [
            self.q_weights_cpu,
            self.k_weights_cpu,
            self.v_weights_cpu,
            self.o_weights_cpu,
            self.q_biases_cpu,
            self.k_biases_cpu,
            self.v_biases_cpu,
            self.o_biases_cpu,
        ]
        for w in _weights:
            self.nparams += w.size

        # Dropout Descriptor
        states_size = cudnn.cudnnDropoutGetStatesSize(self.model.cudnn_handle)
        self.states = TensorArray.zeros(
            (states_size,),
            self.model.dtype,
            tensor_format=self.model.tensor_format,
            mem_type=MemoryType.GLOBAL if self.model.use_gpudirect else MemoryType.DEVICE,
            desc_type=DescriptorType.NONE
        )
        self.memory_used += self.states.nbytes

        self.drop_desc = cudnn.cudnnCreateDropoutDescriptor()
        cudnn.cudnnSetDropoutDescriptor(
            self.drop_desc,
            self.model.cudnn_handle,
            self.dropout_rate,
            self.states.ptr_voidp,
            self.states.size,
            seed=self.model.random_seed,
        )

        # Attention Descriptor
        self.attn_mode = cudnn.cudnnAttnMode["CUDNN_ATTN_QUERYMAP_ONE_TO_ONE"]
        self.attn_mode += cudnn.cudnnAttnMode["CUDNN_ATTN_ENABLE_PROJ_BIASES"]
        self.add_grad = cudnn.cudnnWgradMode["CUDNN_WGRAD_MODE_SET"]
        self.attn_desc = cudnn.cudnnCreateAttnDescriptor()
        cudnn.cudnnSetAttnDescriptor(
            self.attn_desc,
            self.attn_mode,
            self.heads,
            1.0,
            self.model.cudnn_dtype,
            self.model.cudnn_dtype,
            cudnn.cudnnMathType["CUDNN_DEFAULT_MATH"],
            0,  # NOTE: We have a Dropout but we don't use it because the author of this layer didn't use it
            0,  # NOTE: We have a Dropout but we don't use it because the author of this layer didn't use it
            self.embedl,
            self.embedl,
            self.embedl,
            self.d_k,
            self.d_k,
            self.d_k,
            self.embedl,
            self.seq,
            self.seq,
            self.model.batch_size,
            self.beam,
        )

        # GPU Memory Allocation
        self.weights_size, self.workspace_size, self.reserve_backward_size = (
            cudnn.cudnnGetMultiHeadAttnBuffers(self.model.cudnn_handle, self.attn_desc)
        )
        self.model.backend.check_convolution_memory(self.workspace_size)
        weights_size = self.weights_size // np.dtype(self.model.dtype).itemsize
        reserve_backward_size = (
            self.reserve_backward_size // np.dtype(self.model.dtype).itemsize + 1
        )
        self.weights = TensorArray.zeros(
            (weights_size,),
            self.model.dtype,
            tensor_format=self.model.tensor_format,
            mem_type=MemoryType.GLOBAL if self.model.use_gpudirect else MemoryType.DEVICE,
            desc_type=DescriptorType.NONE
        )
        self.memory_used += self.weights.nbytes
        self.dw = TensorArray.zeros(
            (weights_size,),
            self.model.dtype,
            tensor_format=self.model.tensor_format,
            mem_type=MemoryType.GLOBAL if self.model.use_gpudirect else MemoryType.DEVICE,
            desc_type=DescriptorType.NONE
        )
        self.memory_used += self.dw.nbytes
        self.reserve_backward = TensorArray.zeros(
            (reserve_backward_size,),
            self.model.dtype,
            tensor_format=self.model.tensor_format,
            mem_type=MemoryType.GLOBAL if self.model.use_gpudirect else MemoryType.DEVICE,
            desc_type=DescriptorType.NONE
        )
        self.memory_used += self.reserve_backward.nbytes

        # Weights to GPU
        # self.copy_weights()
        # _weights_types = [
        #     "CUDNN_MH_ATTN_Q_WEIGHTS",
        #     "CUDNN_MH_ATTN_K_WEIGHTS",
        #     "CUDNN_MH_ATTN_V_WEIGHTS",
        #     "CUDNN_MH_ATTN_O_WEIGHTS",
        #     "CUDNN_MH_ATTN_Q_BIASES",
        #     "CUDNN_MH_ATTN_K_BIASES",
        #     "CUDNN_MH_ATTN_V_BIASES",
        #     "CUDNN_MH_ATTN_O_BIASES",
        # ]
        _weights = [
            self.q_weights_cpu,
            self.k_weights_cpu,
            self.v_weights_cpu,
            self.o_weights_cpu,
            self.q_biases_cpu,
            self.k_biases_cpu,
            self.v_biases_cpu,
            self.o_biases_cpu,
        ]
        drv.memcpy_htod(
            self.weights.ptr_voidp.value, np.concatenate([w.flatten() for w in _weights])
        )

        # Memory Allocation for Outputs
        self.y = TensorArray.zeros(
            (self.model.batch_size, self.beam, self.seq, self.embedl),
            self.model.dtype,
            tensor_format=self.model.tensor_format,
            mem_type=MemoryType.GLOBAL if self.model.use_gpudirect else MemoryType.DEVICE,
            desc_type=DescriptorType.SEQ
        )
        self.memory_used += self.y.nbytes
        self.dquery = TensorArray.zeros(
            (self.model.batch_size, self.beam, self.seq, self.embedl),
            self.model.dtype,
            tensor_format=self.model.tensor_format,
            mem_type=MemoryType.GLOBAL if self.model.use_gpudirect else MemoryType.DEVICE,
            desc_type=DescriptorType.SEQ
        )
        self.memory_used += self.dquery.nbytes
        self.dkey = TensorArray.zeros(
            (self.model.batch_size, self.beam, self.seq, self.embedl),
            self.model.dtype,
            tensor_format=self.model.tensor_format,
            mem_type=MemoryType.GLOBAL if self.model.use_gpudirect else MemoryType.DEVICE,
            desc_type=DescriptorType.SEQ
        )
        self.memory_used += self.dkey.nbytes
        self.dvalue = TensorArray.zeros(
            (self.model.batch_size, self.beam, self.seq, self.embedl),
            self.model.dtype,
            tensor_format=self.model.tensor_format,
            mem_type=MemoryType.GLOBAL if self.model.use_gpudirect else MemoryType.DEVICE,
            desc_type=DescriptorType.SEQ
        )
        self.memory_used += self.dvalue.nbytes

        self.current_index = -1  # Training
        self.low_window_index = np.full(
            shape=(self.batch, self.beam, self.seq), fill_value=0, dtype=np.int32
        )
        self.high_window_index = np.full(
            shape=(self.batch, self.beam, self.seq), fill_value=self.seq, dtype=np.int32
        )
        # self.dev_seq_lengths_QO = np.full(shape=(self.batch*self.beam), fill_value=self.seq, dtype=np.int32)
        self.dev_seq_lengths_qo = TensorArray.asarray(
            self.y.seq_length_array,
            mem_type=MemoryType.GLOBAL if self.model.use_gpudirect else MemoryType.DEVICE,
            desc_type=DescriptorType.NONE
        )
        self.memory_used += self.dev_seq_lengths_qo.nbytes
        # self.dev_seq_lengths_KV = np.full(shape=(self.batch*self.beam), fill_value=self.seq, dtype=np.int32)
        self.dev_seq_lengths_kv = TensorArray.asarray(
            self.dkey.seq_length_array,
            mem_type=MemoryType.GLOBAL if self.model.use_gpudirect else MemoryType.DEVICE,
            desc_type=DescriptorType.NONE
        )
        self.memory_used += self.dev_seq_lengths_kv.nbytes
        # for i in range(len(_weights)):
        #     wDesc, dest = cudnn.cudnnGetMultiHeadAttnWeights(
        #         self.model.cudnn_handle,
        #         self.attn_desc,
        #         cudnn.cudnnMultiHeadAttnWeightKind[_weights_types[i]],
        #         self.weights_size,
        #         self.weights.ptr_voidp,
        #     )
        #     print(_weights_types[i], dest)
        #     # Check wDesc order matches
        #     if dest is not None:
        #         pycuda.driver.memcpy_htod(dest[0], _weights[i])

    def forward(
        self,
        query: TensorArray,
        key: TensorArray,
        value: TensorArray,
        mask: TensorArray | None = None,
        residuals: TensorArray | None = None,
    ) -> TensorArray:
        """Performs the forward pass of the Multi-Head Attention layer."""
        if True:  # self.model.mode == Mode.TRAIN:
            self.query, self.key, self.value = query, key, value
            # return self.query
            assert residuals
            cudnn.cudnnMultiHeadAttnForward(
                self.model.cudnn_handle,
                self.attn_desc,
                self.current_index,
                self.low_window_index.ctypes.data_as(ctypes.c_void_p),
                self.high_window_index.ctypes.data_as(ctypes.c_void_p),
                self.dev_seq_lengths_qo.ptr_voidp,
                self.dev_seq_lengths_kv.ptr_voidp,
                self.dquery.desc,
                query.ptr_voidp,
                residuals.ptr_voidp,
                self.dkey.desc,
                key.ptr_voidp,
                self.dvalue.desc,
                value.ptr_voidp,
                self.y.desc,
                self.y.ptr_voidp,
                self.weights_size,
                self.weights.ptr_voidp,
                self.model.backend.get_convolution_workspace_size(),
                self.model.backend.get_convolution_workspace_ptr(),
                self.reserve_backward_size,
                self.reserve_backward.ptr_voidp,
            )
        else:
            cudnn.cudnnMultiHeadAttnForward(
                self.model.cudnn_handle,
                self.attn_desc,
                self.current_index,
                self.low_window_index,
                self.high_window_index,
                self.dev_seq_lengths_qo.ptr_voidp,
                self.dev_seq_lengths_kv.ptr_voidp,
                self.dquery.desc,
                query.ptr_voidp,
                residuals.ptr_voidp,
                self.dkey.desc,
                key.ptr_voidp,
                self.dvalue.desc,
                value.ptr_voidp,
                self.y.desc,
                self.y.ptr_voidp,
                self.weights_size,
                self.weights.ptr_voidp,
                self.model.backend.get_convolution_workspace_size(),
                self.model.backend.get_convolution_workspace_ptr(),
                0,
                None,
            )
        return self.y

    def backward(self, dy: TensorArray) -> tuple[TensorArray, TensorArray, TensorArray]:
        """Performs the backward pass to compute gradients for data and weights."""
        cudnn.cudnnMultiHeadAttnBackwardData(
            self.model.cudnn_handle,
            self.attn_desc,
            self.low_window_index.ctypes.data_as(ctypes.c_void_p),
            self.high_window_index.ctypes.data_as(ctypes.c_void_p),
            self.dev_seq_lengths_qo.ptr_voidp,
            self.dev_seq_lengths_kv.ptr_voidp,
            self.y.desc,
            dy.ptr_voidp,
            self.dquery.desc,
            self.dquery.ptr_voidp,
            self.query.ptr_voidp,
            self.dkey.desc,
            self.dkey.ptr_voidp,
            self.key.ptr_voidp,
            self.dvalue.desc,
            self.dvalue.ptr_voidp,
            self.value.ptr_voidp,
            self.weights_size,
            self.weights.ptr_voidp,
            self.model.backend.get_convolution_workspace_size(),
            self.model.backend.get_convolution_workspace_ptr(),
            self.reserve_backward_size,
            self.reserve_backward.ptr_voidp,
        )

        cudnn.cudnnMultiHeadAttnBackwardWeights(
            self.model.cudnn_handle,
            self.attn_desc,
            self.add_grad,
            self.dquery.desc,
            self.query.ptr_voidp,
            self.dkey.desc,
            self.key.ptr_voidp,
            self.dvalue.desc,
            self.value.ptr_voidp,
            self.y.desc,
            dy.ptr_voidp,
            self.weights_size,
            self.weights.ptr_voidp,
            self.dw.ptr_voidp,
            self.model.backend.get_convolution_workspace_size(),
            self.model.backend.get_convolution_workspace_ptr(),
            self.reserve_backward_size,
            self.reserve_backward.ptr_voidp,
        )

        return self.dquery, self.dkey, self.dvalue
