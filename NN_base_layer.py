"""
Base Layer definition for Python Distributed Training of Neural Networks (PyDTNN)

PyDTNN is a light-weight library for distributed Deep Learning training and
inference that offers an initial starting point for interaction with distributed
training of (and inference with) deep neural networks. PyDTNN prioritizes
simplicity over efficiency, providing an amiable user interface which enables a
flat accessing curve. To perform the training and inference processes, PyDTNN
exploits distributed inter-process parallelism (via MPI) for clusters and
intra-process (via multi-threading) parallelism to leverage the presence of
multicore processors and GPUs at node level. For that, PyDTNN uses MPI4Py for
message-passing, BLAS calls via NumPy for multicore processors and
PyCUDA+cuDNN+cuBLAS for NVIDIA GPUs.

Copyright 2020 Universitat Jaume I

This file is part of PyDTNN. PyDTNN is free software: you can redistribute it
and/or modify it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

PyDTNN is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details. You
should have received a copy of the GNU General Public License along with this
program. If not, see <http://www.gnu.org/licenses/>.
"""

__author__ = "Manuel F. Dolz, Enrique S. Quintana, \
              Mar Catalan, Adrian Castello"
__contact__ = "dolzm@uji.es"
__copyright__ = "Copyright 2020, Universitat Jaume I"
__credits__ = ["Manuel F. Dolz, Enrique S. Quintana", \
               "Mar Catalan", "Adrian Castello"]
__date__ = "2020/03/22"

__email__ = "dolzm@uji.es"
__license__ = "GPLv3"
__maintainer__ = "Manuel F. Dolz"
__status__ = "Production"
__version__ = "1.1.0"

import importlib
from math import floor

# import NN_activation
import NN_initializer
import NN_model
from NN_conv_gemm import ConvGemm
from NN_add_cython import add_cython
from NN_argmax_cython import argmax_cython
from NN_im2col_cython import im2col_cython, col2im_cython
from NN_sim import *
from NN_tracer import PYDTNN_MDL_EVENT, PYDTNN_MDL_EVENTS, PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_MDL_ALLREDUCE_DW, \
    PYDTNN_OPS_ALLREDUCE_DW

try:
    from mpi4py import MPI
    # import pycuda.autoinit
    import pycuda.gpuarray as gpuarray
    import pycuda.driver as drv
    import skcuda.linalg as culinalg
    import skcuda.misc as cumisc
    import libnccl.libnccl as nccl
except ModuleNotFoundError:
    pass
except ImportError:
    pass


class Layer:

    def __new__(cls, *args, **kwargs):
        # If GPU is requested, return a GPU-related object instead
        if not NN_model.enable_cudnn:
            new_cls = cls
        else:
            module_name = "NN_activation" if cls.__name__ in \
                                             ["Sigmoid", "Relu", "Tanh", "Log", "Softmax"] else "NN_layer"
            module = importlib.import_module("%s_gpu" % module_name)
            new_cls = getattr(module, f"{cls.__name__}GPU")
        instance = super(Layer, new_cls).__new__(new_cls)
        if new_cls != cls:
            instance.__init__(*args, **kwargs)
        return instance

    def __init__(self, shape=()):
        self.id = 0
        self.nparams = 0
        self.shape = shape
        self.weights = np.array([])
        self.biases = np.array([])
        self.act = None
        self.grad_vars = {}
        self.fwd_time = np.zeros((4,), dtype=np.float32)
        self.bwd_time = np.zeros((4,), dtype=np.float32)
        self.paths = []
        # The next attributes will be initialized later
        self.model = None
        self.prev_shape = None
        self.need_dx = True
        self.is_block_layer = False

    def initialize(self, prev_shape, need_dx=True):
        self.prev_shape = prev_shape
        self.need_dx = need_dx

    def show(self, attrs=""):
        if not attrs:
            attrs = "|{:19s}|{:^24s}|".format("", "")
        print(f"|{self.id:^7d}|{type(self).__name__:^26s}|{self.nparams:^9d}|{str(self.shape):^15}" + attrs)

    @property
    def children(self):
        children = []
        for path in self.paths:
            children += [layer for layer in path]
        return children

    def update_weights(self, optimizer):
        optimizer.update(self)

    def reduce_weights_async(self):
        if not self.model.comm:
            return
        self.reqs_allred = {}

        # if self.model.enable_cudnn:
        #     if self.model.enable_nccl or self.model.gpudirect:
        #        self.model.stream.synchronize()
        #     else:
        #        self.stream_2.synchronize()

        for w_, dw_ in self.grad_vars.items():
            dw = getattr(self, dw_)

            if self.model.enable_cudnn:
                if self.model.enable_nccl:
                    self.model.stream.synchronize()
                    nccl.ncclAllReduce(dw.ptr, dw.ptr, dw.size, self.model.nccl_type,
                                       nccl.RedOp.Sum, comm=self.model.nccl_comm,
                                       stream=self.stream_2.handle)
                    req = None

                    # # Hierarchical mode NCCL + MPI
                    # if len(self.model.inter_ranks) == 1:
                    #     nccl.ncclAllReduce(dw.ptr, dw.ptr, dw.size, self.model.nccl_type, 
                    #                        nccl.RedOp.Sum, comm=self.model.nccl_comm, 
                    #                        stream=self.stream_2.handle)
                    #
                    # else:
                    #     # Hierarchical allreduce - Phase 1: ncclReduce + Iallreduce
                    #     nccl.ncclReduce(dw.ptr, dw.ptr, dw.size, self.model.nccl_type, 
                    #                     nccl.RedOp.Sum, root=0, comm=self.model.nccl_comm, 
                    #                     stream=self.stream_2.handle)
                    #
                    #     if self.model.rank in self.model.inter_ranks:
                    #         if not self.model.gpudirect:
                    #             dw.ary.get_async(self.stream_2, dw_cpu)
                    #
                    #         self.stream_2.synchronize()
                    #         req = self.model.inter_comm.Iallreduce(MPI.IN_PLACE, dw_cpu, op=MPI.SUM) 

                else:  # Without NCCL

                    # We have asynchronusly moved the dw and db to dw_cpu and db_cpu in stream_2
                    # so we need to synchronize stream_2 before performing Allreduce.
                    # In GPU direct we have to synchronize the main stream to ensure dw and db are ready.

                    if not self.model.gpudirect:
                        self.stream_2.synchronize()
                    else:
                        self.model.stream.synchronize()

                    dw_cpu = getattr(self, "%s_cpu" % dw_)
                    req = self.model.comm.Iallreduce(MPI.IN_PLACE, dw_cpu, op=MPI.SUM)

            else:  # Without GPUs, only MPI
                req = self.model.comm.Iallreduce(MPI.IN_PLACE, dw, op=MPI.SUM)

            self.reqs_allred[dw_] = req

    def wait_allreduce_async(self):
        if not self.model.comm or self.model.enable_nccl:
            return

        for w_, dw_ in self.grad_vars.items():
            self.reqs_allred[dw_].wait()

            # # Hierarchical mode NCCL + MPI
            # if self.model.enable_nccl:  
            #     if len(self.model.inter_ranks) == 1: 
            #         # Do nothing, Allreduce was already completed in phase 1
            #         pass
            #     else:
            #         # Hierarchical allreduce - Phase 2: wait + ncclBroadcast
            #         if self.model.rank in self.model.inter_ranks:
            #             self.reqs_allred[dw_].wait()
            #             if not self.model.gpudirect: 
            #                 dw.ary.set_async(dw_cpu, self.stream_2)
            #     
            #         nccl.ncclBroadcast(dw.ptr, dw.ptr, dw.size, self.model.nccl_type, 
            #                            root=0, comm=self.model.nccl_comm, 
            #                            stream=self.stream_2.handle)

            if self.model.enable_cudnn and not self.model.gpudirect:
                dw = getattr(self, dw_)
                dw_cpu = getattr(self, "%s_cpu" % dw_)

                # If there is no CUDA-aware MPI, copy data back to GPU
                dw.ary.set_async(dw_cpu, self.stream_2)

    def reduce_weights_sync(self):
        if not self.model.comm: return

        for w_, dw_ in self.grad_vars.items():
            self.tracer.emit_nevent([PYDTNN_MDL_EVENT, PYDTNN_OPS_EVENT],
                                    [self.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_ALLREDUCE_DW,
                                     self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_ALLREDUCE_DW])
            dw = getattr(self, dw_)

            if self.model.enable_cudnn:

                if self.model.enable_nccl:
                    nccl.ncclAllReduce(dw.ptr, dw.ptr, dw.size, self.model.nccl_type,
                                       nccl.RedOp.Sum, comm=self.model.nccl_comm,
                                       stream=self.stream_2.handle)

                    # # Hierarchical mode NCCL + MPI
                    # if len(self.model.inter_ranks) == 1:
                    #     # Only one node involved, perform ncclAllreduce across intra-node GPUs
                    #     nccl.ncclAllReduce(dw.ptr, dw.ptr, dw.size, self.model.nccl_type, 
                    #                        nccl.RedOp.Sum, comm=self.model.nccl_comm, 
                    #                        stream=self.stream_2.handle)
                    # else:
                    #     # Hierarchical allreduce: ncclReduce + Allreduce + ncclBroadcast
                    #     nccl.ncclReduce(dw.ptr, dw.ptr, dw.size, self.model.nccl_type, 
                    #                     nccl.RedOp.Sum, root=0, comm=self.model.nccl_comm,
                    #                     stream=self.stream_2.handle)
                    # 
                    #     self.stream_2.synchronize()
                    #     if self.model.rank in self.model.inter_ranks:
                    #         if self.model.gpudirect: 
                    #             self.model.inter_comm.Allreduce(MPI.IN_PLACE, dw_cpu, op=MPI.SUM) 
                    #         else:
                    #             dw_cpu = dw.ary.get()
                    #             self.model.inter_comm.Allreduce(MPI.IN_PLACE, dw_cpu, op=MPI.SUM)
                    #             dw.ary.set_async(dw_cpu, self.stream_2)
                    # 
                    #     nccl.ncclBroadcast(dw.ptr, dw.ptr, dw.size, self.model.nccl_type, 
                    #                        root=0, comm=self.model.nccl_comm, 
                    #                        stream=self.stream_2.handle)

                else:  # Without NCCL

                    # We have asynchronusly moved the dw and db to dw_cpu and db_cpu in stream_2
                    # so we need to synchronize stream_2 before performing Allreduce.
                    # In GPU direct, the main stream is already synchronized.

                    if not self.model.gpudirect:
                        self.stream_2.synchronize()

                    dw_cpu = getattr(self, "%s_cpu" % dw_)
                    self.model.comm.Allreduce(MPI.IN_PLACE, dw_cpu, op=MPI.SUM)

                    if not self.model.gpudirect:
                        dw.ary.set_async(dw_cpu, self.stream_2)
            else:
                self.model.comm.Allreduce(MPI.IN_PLACE, dw, op=MPI.SUM)

            self.tracer.emit_nevent([PYDTNN_MDL_EVENT, PYDTNN_OPS_EVENT], [0, 0])
