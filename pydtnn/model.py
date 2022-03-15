"""
PyDTNN model
"""

#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021-22 Universitat Jaume I
#
#  PyDTNN is free software: you can redistribute it and/or modify it under the
#  terms of the GNU General Public License as published by the Free Software
#  Foundation, either version 3 of the License, or (at your option) any later
#  version.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT
#  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
#  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
#  License for more details.
#
#  You should have received a copy of the GNU General Public License along
#  with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import functools
import importlib
import os
import sys
import time
from timeit import default_timer as timer
from typing import Any

from tqdm import tqdm

import pydtnn.metrics
from pydtnn.utils import PYDTNN_TENSOR_FORMAT_NHWC, PYDTNN_TENSOR_FORMAT_NCHW
from . import losses, metrics
from . import utils
from .datasets import CustomDataset, get_dataset
from .lr_schedulers import get_lr_schedulers
from .optimizers import get_optimizer
from .parser import parser
from .performance_models import *
from .tracers import PYDTNN_MDL_EVENT, PYDTNN_MDL_EVENTS, PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, ExtraeTracer, \
    SimpleTracer, PYDTNN_MDL_UPDATE_DW, PYDTNN_OPS_ALLREDUCE_DW, PYDTNN_MDL_WAIT_DW, \
    PYDTNN_MDL_FORWARD, PYDTNN_MDL_BACKWARD, PYDTNN_MDL_ALLREDUCE_DW
from .utils.best_of import BestOf
from .utils.memory_cache import MemoryCache
from .utils.performance_counter import PerformanceCounter

supported_gpu = False
supported_cudnn = True
supported_nccl = True
supported_mpi4py = True
enable_cudnn = False
gpuarray: Any = None

try:
    # noinspection PyUnresolvedReferences,PyPackageRequirements
    from mpi4py import MPI
except (ImportError, ModuleNotFoundError):
    supported_mpi4py = False

EVALUATE_MODE, TRAIN_MODE, UNSPECIFIED_MODE = (0, 1, 2)


def _layer_id_generator():
    """To obtain consecutive layer ids. See Layer.set_model()."""
    current_layer_id = 0
    while True:
        yield current_layer_id
        current_layer_id += 1


def ensure_model_is_initialized(method):
    @functools.wraps(method)
    def wrapper_ensure_model_is_initialized(*args, **kwargs):
        self = args[0]
        if not self._initialized:
            self._initialize()
        return method(*args, **kwargs)

    return wrapper_ensure_model_is_initialized


class Model:
    """
    PyDTNN Model
    """

    def __init__(self, parallel="sequential", non_blocking_mpi=False, enable_gpu=False, enable_gpudirect=False,
                 enable_nccl=False, dtype=np.float32, tracing=False, tracer_output="", **kwargs):
        # Attributes related to the given arguments
        self.parallel = parallel
        self.blocking_mpi = not non_blocking_mpi
        global enable_cudnn
        enable_cudnn = self.enable_cudnn = enable_gpu
        self.gpudirect = enable_gpudirect
        self.enable_nccl = enable_nccl
        self.dtype = dtype
        # Set MPI and comm
        if self.parallel == "sequential":
            self.MPI = None
            self.comm = None
        elif self.parallel == "data":
            if not supported_mpi4py:
                raise SystemExit("Please, install mpi4py to allow parallel MPI execution!")
            self.MPI = MPI
            self.comm = MPI.COMM_WORLD
        else:
            raise SystemExit(f"Parallel option '{parallel}' not recognized.")
        # Set tracer
        if tracer_output == "" and not enable_gpu:
            self.tracer = ExtraeTracer(tracing)
        elif enable_gpu:
            from .tracers import SimpleTracerGPU
            self.tracer = SimpleTracerGPU(tracing, tracer_output, self.comm)
        else:
            self.tracer = SimpleTracer(tracing, tracer_output, self.comm)
        # Get default values from parser and update them from the received kwargs
        self.kwargs = vars(parser.parse_args([]))
        self.kwargs.update(kwargs)
        # Set performance counter
        self.perf_counter = PerformanceCounter()
        # Layers' attributes
        self.layers = []
        self.layer_id = _layer_id_generator()
        # In data parallel, we assume that file weights are stored in a nfs mounted directory.
        self.shared_storage = True
        # Matmul
        self.matmul = getattr(utils, "matmul")
        # Set current mode to unspecified
        self.mode = UNSPECIFIED_MODE
        # Memory cache optimization
        if self.enable_memory_cache:
            MemoryCache.enable()
        else:
            MemoryCache.disable()
        # Initialize the total number of params of the model
        self.nparams = 0
        # Execution attributes
        self.rank = 0
        self.nprocs = 1
        if self.comm:
            if supported_mpi4py:
                self.rank = self.comm.Get_rank()
                self.nprocs = self.comm.Get_size()
            else:
                raise SystemExit("Please, install mpi4py to allow parallel MPI execution!")
        if self.enable_cudnn:
            global supported_cudnn, supported_nccl
            supported_cudnn = True
            supported_nccl = True
            try:
                import pydtnn.backends.gpu.tensor_gpu
                global gpuarray
                # noinspection PyUnresolvedReferences
                import pycuda.gpuarray as gpuarray
                # noinspection PyUnresolvedReferences
                import pycuda.driver as drv
                from pydtnn.backends.gpu.libs import libcudnn as cudnn
                # noinspection PyUnresolvedReferences
                from skcuda import cublas
            except (ImportError, ModuleNotFoundError, OSError):
                msg = "Please, install pycuda, skcuda, and cudnn to be able to use the GPUs!"
                raise SystemExit(msg) from None

            # import pycuda.autoinit
            # The next fake test exists only to avoid the pycuda.autoinit import being removed when optimizing imports
            # if self.kwargs.get('fake_pycuda_autoinit_option'):
            #    pycuda.autoinit()
            # Uncomment the next code if pycuda.autoinit is not available
            device_id = self.rank % drv.Device.count()
            drv.init()
            context = drv.Device(device_id).make_context()
            import atexit
            atexit.register(context.pop)

            if self.comm and self.enable_nccl:
                try:
                    from pydtnn.backends.gpu.libs import libnccl as nccl
                except (ImportError, ModuleNotFoundError, OSError):
                    supported_nccl = False
                    msg = "Please, install nccl to be able to use NVIDIA NCCL inter-GPU communications!"
                    raise SystemExit(msg) from None

                types = {np.float64: nccl.DataType.Float64,
                         np.float32: nccl.DataType.Float32,
                         np.int8: nccl.DataType.Int8,
                         np.int32: nccl.DataType.Int32}

                self.nccl_type = types.get(self.dtype, nccl.DataType.Float32)

                hostname = MPI.Get_processor_name()

                hosts_data = self.comm.allgather([self.rank, hostname])
                # Build a dictionary hostname : [ranks_in_host]
                #   { "host1": [0, 1], "host2": [2, 3] }
                hosts = {}
                for r, h in hosts_data:
                    # noinspection PyTypeChecker
                    hosts.setdefault(h, []).append(r)
                if self.parallel == "data":
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(self.rank % self.gpus_per_node)
                # Check that no more processes than GPUs per node are used
                for host, ranks_in_host in hosts.items():
                    if len(ranks_in_host) > self.gpus_per_node:
                        raise SystemExit("Not able to run more processes than GPUs per node!")

                nccl_id = self.comm.bcast(nccl.ncclGetUniqueId() if self.rank == 0 else None)
                self.nccl_comm = nccl.ncclCommInitRank(self.nprocs, nccl_id, self.rank)

                # if self.enable_nccl_hierarchical:
                #     self.intra_ranks = hosts[hostname]
                #     # Only a master process per node is selected as inter rank
                #     self.inter_ranks = [r[0] for h, r in hosts.items()]
                #     
                #     intra_group_ = comm.Get_group()
                #     intra_group = MPI.Group.Incl(intra_group_, self.intra_ranks)
                #     intra_comm = comm.Create(intra_group)
                #     
                #     if len(self.inter_ranks) > 1:
                #         inter_group_ = comm.Get_group()
                #         inter_group = MPI.Group.Incl(inter_group_, self.inter_ranks)
                #         self.inter_comm = comm.Create(inter_group)
                # 
                #     # Get an id once per master process and distribute it to all the intra ranks
                #     id = intra_comm.bcast(nccl.ncclGetUniqueId() if self.rank in self.inter_ranks else None)
                #     self.nccl_comm = nccl.ncclCommInitRank(len(self.intra_ranks), id, intra_comm.Get_rank())

            self.cudnn_handle = cudnn.cudnnCreate()
            self.cublas_handle = cublas.cublasCreate()
            self.stream = drv.Stream()
            cublas.cublasSetStream(self.cublas_handle, self.stream.handle)
            cudnn.cudnnSetStream(self.cudnn_handle, self.stream.handle)

            types = {np.float64: "CUDNN_DATA_DOUBLE",
                     np.float32: "CUDNN_DATA_FLOAT",
                     np.int8: "CUDNN_DATA_INT8",
                     np.int32: "CUDNN_DATA_INT32"}

            cudnn_type = types.get(self.dtype, "CUDNN_DATA_FLOAT")

            self.cudnn_dtype = cudnn.cudnnDataType[cudnn_type]
            self.tracer.set_default_stream(self.stream)
        # Set data format
        if self.tensor_format == "AUTO":
            if self.enable_cudnn:
                self.tensor_format = PYDTNN_TENSOR_FORMAT_NCHW
            else:
                self.tensor_format = PYDTNN_TENSOR_FORMAT_NHWC
        elif self.tensor_format == "NCHW":
            self.tensor_format = PYDTNN_TENSOR_FORMAT_NCHW
        else:
            self.tensor_format = PYDTNN_TENSOR_FORMAT_NHWC
        # Disable BestOf globally if not enabled
        if self.kwargs['enable_best_of'] is False:
            BestOf.use_always_the_first_alternative()
        # Explicit declaration of those model attributes that are referenced by other parts of PyDTNN
        self.batch_size = self.kwargs['batch_size']
        self.steps_per_epoch = self.kwargs['steps_per_epoch']
        self.cpu_speed = self.kwargs['cpu_speed']
        self.memory_bw = self.kwargs['memory_bw']
        self.network_bw = self.kwargs['network_bw']
        self.network_lat = self.kwargs['network_lat']
        self.network_alg = self.kwargs['network_alg']
        # Load weights and bias
        if self.weights_and_bias_filename:
            self.load_weights_and_bias(self.weights_and_bias_filename)
        # Dataset
        self.dataset = get_dataset(self)
        # Optimizers and LRSchedulers
        if self.kwargs["optimizer_name"] == "sgd" and self.kwargs["learning_rate_scaling"]:
            self.kwargs["learning_rate"] *= self.nprocs
        self.optimizer = get_optimizer(self)
        self.lr_schedulers = get_lr_schedulers(self)
        # Metrics list
        self.metrics_list = [m for m in self.metrics.replace(" ", "").split(",")]
        # Private attributes
        self._evaluate_round = 0
        self._initialized = False
        # Attributes that will be properly defined elsewhere
        self.y_batch = None
        self.history = None
        # Read the model (must be the last action, as it calls self._initialize() if there is a model)
        self.model_name = self.kwargs.get("model_name")
        if self.model_name:
            self._read_model(self.model_name)

    def __getattr__(self, item):
        try:
            return self.kwargs[item]
        except KeyError:
            raise AttributeError(f"Model object has no attribute '{item}'!") from None

    def _read_model(self, model_name):
        try:
            model_module = importlib.import_module(f"pydtnn.models.{model_name}")
            getattr(model_module, f"create_{model_name}")(self)
        except (ModuleNotFoundError, AttributeError):
            import traceback
            print(traceback.format_exc())
            sys.exit(-1)
        else:  # There was no error, call _initialize()
            self._initialize()

    def show(self):
        bfp = {np.float32: 4, np.float64: 8}[self.dtype]
        line = "+-------+--------------------------+---------+---------------+-------------------" \
               "+-------------------------------------+"
        head = "| Layer |           Type           | #Params | Output shape  |   Weights shape   " \
               "|             Parameters              |"
        print(line)
        print(head)
        for layer in self.layers:
            print(line)
            layer.show()
        print(line)
        print(f"|{'':^7s} {'Total parameters':^26s} {self.nparams:^9d} {utils.convert_size(self.nparams * bfp):^15s} "
              f"{'':19s} {'':37s}|")
        print(line)

    def print_in_convdirect_format(self):
        line = "#l\tkn\two\tho\tt\tkh\tkw\tci\twi\thi"
        print(line)
        for layer in self.layers:
            layer.print_in_convdirect_format()

    def add(self, layer):
        layer.set_model(self)
        need_dx = layer.id > 1
        prev_shape = self.layers[-1].shape if layer.id > 0 else ()

        if self.enable_cudnn:
            y = self.layers[-1].y if layer.id > 0 else None
            layer.initialize(prev_shape, need_dx, y)
        else:
            layer.initialize(prev_shape, need_dx)

        self.nparams += layer.nparams
        self.layers.append(layer)

        if layer.act:
            self.add(layer.act())

    def get_all_layers(self, from_layers=None):
        if from_layers is None:
            from_layers = self.layers
        this_recursion_layers = []
        for layer in from_layers:
            this_recursion_layers.append(layer)
            this_recursion_layers += self.get_all_layers(layer.children)
        return this_recursion_layers

    def _apply_layer_fusion(self, bn_relu=False, conv_relu=False, conv_bn=False, conv_bn_relu=False):
        """ Apply layer fusion in a recursive manner """

        def __layer_fusion(layers, bn_relu=False, conv_relu=False, conv_bn=False, conv_bn_relu=False):
            fused_layers = []
            for i, curr_layer in enumerate(layers):
                # if i > 0: print(i, curr_layer.canonical_name, fused_layers[-1].canonical_name)
                if curr_layer.is_block_layer:
                    for j, p in enumerate(curr_layer.paths):
                        curr_layer.paths[j] = __layer_fusion(p, bn_relu, conv_relu, conv_bn, conv_bn_relu)
                elif conv_bn_relu and len(fused_layers) > 1 and \
                        curr_layer.canonical_name == "Relu" and \
                        fused_layers[-1].canonical_name == "BatchNormalization" and \
                        fused_layers[-2].canonical_name == "Conv2D":
                    backend = "gpu" if self.enable_cudnn else "cpu"
                    fused_layer = getattr(importlib.import_module(f"pydtnn.backends.{backend}.layers"),
                                          fused_layers[-2].canonical_name +
                                          fused_layers[-1].canonical_name +
                                          type(curr_layer).__name__)
                    if fused_layers[-2].forward.__name__ in fused_layer.__dict__:  # or self.enable_best_of:
                        bn_layer = fused_layers.pop()
                        cv_layer = fused_layers.pop()
                        print("Fusing %03d_%s + %03d_%s + %03d_%s..." % (cv_layer.id, type(cv_layer).__name__,
                                                                         bn_layer.id, type(bn_layer).__name__,
                                                                         curr_layer.id, type(curr_layer).__name__))
                        curr_layer = fused_layer(from_parent=cv_layer, from_parent2=bn_layer)
                        curr_layer.initialize(from_parent_dict=cv_layer.__dict__)
                elif (conv_relu or conv_bn) and len(fused_layers) > 0 and \
                        (curr_layer.canonical_name == "Relu" or
                         curr_layer.canonical_name == "BatchNormalization") and \
                        fused_layers[-1].canonical_name == "Conv2D" and \
                        not (conv_bn_relu and i + 1 < len(layers) and layers[i + 1].canonical_name == "Relu"):
                    backend = "gpu" if self.enable_cudnn else "cpu"
                    fused_layer = getattr(importlib.import_module(f"pydtnn.backends.{backend}.layers"),
                                          fused_layers[-1].canonical_name +
                                          type(curr_layer).__name__)
                    if fused_layers[-1].forward.__name__ in fused_layer.__dict__:  # or self.enable_best_of:
                        prev_layer = fused_layers.pop()
                        print("Fusing %03d_%s + %03d_%s ..." % (prev_layer.id, type(prev_layer).__name__,
                                                                curr_layer.id, type(curr_layer).__name__))
                        curr_layer = fused_layer(from_parent=prev_layer, from_parent2=curr_layer)
                        curr_layer.initialize(from_parent_dict=prev_layer.__dict__)
                elif bn_relu and len(fused_layers) > 0 and \
                        curr_layer.canonical_name == "Relu" and \
                        fused_layers[-1].canonical_name == "BatchNormalization":
                    prev_layer = fused_layers.pop()
                    print("Fusing %03d_%s + %03d_%s ..." % (prev_layer.id, type(prev_layer).__name__,
                                                            curr_layer.id, type(curr_layer).__name__))
                    curr_layer = getattr(importlib.import_module("pydtnn.layers"),
                                         prev_layer.canonical_name +
                                         curr_layer.canonical_name)(from_parent=prev_layer)
                fused_layers.append(curr_layer)
            return fused_layers

        if not self.enable_cudnn and (bn_relu or conv_relu or conv_bn, conv_bn_relu):
            self.layers = __layer_fusion(self.layers, bn_relu, conv_relu, conv_bn, conv_bn_relu)

    def load_store_path(self, layers, d, mode):
        for layer in layers:
            name = layer.canonical_name
            if name in ["AdditionBlock", "ConcatenationBlock"]:
                for path in layer.paths:
                    self.load_store_path(path, d, mode)
            else:
                grad_vars = [g for g in layer.grad_vars] + \
                            (["running_var", "running_mean"] if name == "BatchNormalization" else [])
                if name == "BatchNormalization":
                    layer.updated_running_var = True
                for key in grad_vars:
                    base = f"{layer.id}_{name}_{key}"
                    if mode == "load" and base not in d:
                        print(f"Could not find '{base}' for layer '{name}' in file!")
                        continue
                    if mode == "load":
                        if self.enable_cudnn:
                            ary = getattr(layer, key).ary
                            ary.set(d[base].reshape(ary.shape))
                        else:
                            setattr(layer, key, d[base])
                    elif mode == "store":
                        if self.enable_cudnn:
                            d[base] = getattr(layer, key).ary.get()
                        else:
                            d[base] = getattr(layer, key)

    def load_weights_and_bias(self, filename):
        d = np.load(filename)
        self.load_store_path(self.layers, d, "load")

    def store_weights_and_bias(self, filename):
        if self.shared_storage and self.rank == 0:
            d = {}
            self.load_store_path(self.layers, d, "store")
            np.savez_compressed(filename, **d)

    def calculate_time(self):
        # Total elapsed_time, Comp elapsed_time, Memo elapsed_time, Net elapsed_time
        total_time = np.zeros((4,), dtype=np.float32)

        # Forward pass (FP)
        for layer in range(1, len(self.layers)):
            total_time += self.layers[layer].fwd_time

        if self.blocking_mpi:
            # Blocking MPI
            # Back propagation. Gradient computation (GC) and weights update (WU)
            for layer in range(len(self.layers) - 1, 0, -1):
                total_time += self.layers[layer].bwd_time

            # Weight update (WU)
            for layer in range(len(self.layers) - 1, 0, -1):
                if self.comm and self.layers[layer].weights.size > 0:
                    total_time += allreduce_time(self.layers[layer].weights.size + self.layers[layer].biases.size,
                                                 self.cpu_speed, self.network_bw, self.network_lat,
                                                 self.network_alg, self.nprocs, self.dtype)
        else:
            total_time_iar = 0
            # Non-blocking MPI
            # Back propagation. Gradient computation (GC) and weights update (WU)
            for layer in range(len(self.layers) - 1, 0, -1):
                total_time += self.layers[layer].bwd_time
                if self.comm and self.layers[layer].weights.size > 0:
                    time_iar = allreduce_time(self.layers[layer].weights.size + self.layers[layer].biases.size,
                                              self.cpu_speed, self.network_bw, self.network_lat,
                                              self.network_alg, self.nprocs, self.dtype)
                    total_time[3] += time_iar[3]
                    total_time_iar = max(total_time[0], total_time_iar) + time_iar[0]

            total_time[0] = max(total_time[0], total_time_iar)

        return total_time

    def _compute_metrics_funcs(self, y_pred, y_targ, loss, blocking=True):
        loss_req = None
        if self.enable_cudnn:
            _losses = np.array([loss] + [func(y_pred.ary, y_targ.ary) for func in self.metrics_funcs],
                               dtype=np.float32) / self.nprocs
        else:
            _losses = np.array([loss] + [func(y_pred, y_targ) for func in self.metrics_funcs],
                               dtype=np.float32) / self.nprocs
        if self.comm is not None and blocking:
            self.comm.Allreduce(MPI.IN_PLACE, _losses, op=MPI.SUM)
        elif self.comm is not None and not blocking:
            loss_req = self.comm.Iallreduce(MPI.IN_PLACE, _losses, op=MPI.SUM)
        return _losses, loss_req

    def _update_running_average(self, curr, total, count, batch_size, prefix=""):
        string = ""
        total = ((curr * batch_size) + (total * count)) / (count + batch_size)
        for c in range(len(self.loss_and_metrics)):
            loss_str = pydtnn.metrics.metric_format.get(self.loss_and_metrics[c], self.loss_and_metrics[c])
            string += ("%s, " % (prefix + loss_str)) % total[c]
        string = string[:-2]
        return total, count + batch_size, string

    def _get_x_y_targ(self, x_batch, y_batch, current_batch_size):
        if self.enable_cudnn:
            if x_batch.shape[0] != current_batch_size:
                raise ValueError
            self.layers[0].y.ary.set(x_batch)
            self.y_batch.ary.set(y_batch)
            x, y_targ = self.layers[0].y, self.y_batch
        else:
            x, y_targ = x_batch, y_batch
        return x, y_targ

    def _train_batch(self, x_batch, y_batch, current_batch_size):

        self.mode = TRAIN_MODE
        for lr_sched in self.lr_schedulers:
            lr_sched.on_batch_begin()

        try:
            x, y_targ = self._get_x_y_targ(x_batch, y_batch, current_batch_size)
        except ValueError:
            return self.total_metrics

        # Forward pass (FP)
        for i in range(1, len(self.layers)):
            self.tracer.emit_event(PYDTNN_MDL_EVENT, self.layers[i].id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_FORWARD)
            x = self.layers[i].forward(x)
            self.tracer.emit_event(PYDTNN_MDL_EVENT, 0)

        loss, dx = self.loss_func(x, y_targ, self.batch_size)
        self.total_metrics, _ = self._compute_metrics_funcs(x, y_targ, loss)

        if self.blocking_mpi:
            # Blocking MPI
            # Back propagation. Gradient computation (GC) and weights update (WU)
            for i in range(len(self.layers) - 1, 0, -1):
                # self.tracer.print_memory_usage(f"Layer {l:03} before backward")
                self.tracer.emit_event(PYDTNN_MDL_EVENT, self.layers[i].id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_BACKWARD)
                dx = self.layers[i].backward(dx)
                # self.tracer.print_memory_usage(f"Layer {l:03} after backward ")
                self.tracer.emit_event(PYDTNN_MDL_EVENT, 0)

            if self.enable_cudnn:
                self.stream.synchronize()

            # Weight update (WU)
            for i in range(len(self.layers) - 1, 0, -1):
                self.layers[i].reduce_weights_sync()
                self.tracer.emit_event(PYDTNN_MDL_EVENT, self.layers[i].id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_UPDATE_DW)
                self.layers[i].update_weights(self.optimizer)
                self.tracer.emit_event(PYDTNN_MDL_EVENT, 0)
        else:
            # Non-blocking MPI
            # Back propagation. Gradient computation (GC) and weights update (WU)
            for i in range(len(self.layers) - 1, 0, -1):
                self.tracer.emit_event(PYDTNN_MDL_EVENT, self.layers[i].id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_BACKWARD)
                dx = self.layers[i].backward(dx)
                self.tracer.emit_event(PYDTNN_MDL_EVENT, 0)

                self.tracer.emit_event(PYDTNN_MDL_EVENT,
                                       self.layers[i].id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_ALLREDUCE_DW)
                self.layers[i].reduce_weights_async()
                self.tracer.emit_event(PYDTNN_MDL_EVENT, 0)

            # Weight update (WU)
            for i in range(len(self.layers) - 1, 0, -1):
                self.tracer.emit_nevent([PYDTNN_MDL_EVENT, PYDTNN_OPS_EVENT],
                                        [self.layers[i].id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_WAIT_DW,
                                         self.layers[i].id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_ALLREDUCE_DW])
                self.layers[i].wait_allreduce_async()
                self.tracer.emit_nevent([PYDTNN_MDL_EVENT, PYDTNN_OPS_EVENT], [0, 0])

                self.tracer.emit_event(PYDTNN_MDL_EVENT, self.layers[i].id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_UPDATE_DW)
                self.layers[i].update_weights(self.optimizer)
                self.tracer.emit_event(PYDTNN_MDL_EVENT, 0)

        if self.enable_cudnn:
            for i in range(len(self.layers) - 1, 0, -1):
                if self.layers[i].grad_vars:
                    self.layers[i].stream_2.synchronize()

        for lr_sched in self.lr_schedulers:
            lr_sched.on_batch_end(self)

        return self.total_metrics

    def _initialize(self):
        if self._initialized:
            return
        self._apply_layer_fusion(self.kwargs.get("enable_fused_bn_relu"), self.kwargs.get("enable_fused_conv_relu"),
                                 self.kwargs.get("enable_fused_conv_bn"), self.kwargs.get("enable_fused_conv_bn_relu"))
        self.loss_func = getattr(losses, self.loss_func_name)(shape=(self.batch_size, *self.layers[-1].shape),
                                                              model=self)
        self.metrics_funcs = [getattr(metrics, m)(shape=(self.batch_size, *self.layers[-1].shape), model=self) for m in
                              self.metrics_list]
        self.loss_and_metrics = [self.loss_func_name] + self.metrics_list
        self.tracer.define_event_types(self)
        self._initialized = True

    def train(self, x_train, y_train, x_val, y_val, bar_width=110):
        self.dataset = CustomDataset(self, x_train=x_train, y_train=y_train, x_test=x_val, y_test=y_val)
        history = self.train_dataset(bar_width=bar_width)
        return history

    @ensure_model_is_initialized
    def train_dataset(self, bar_width=110):
        if self.enable_cudnn and self.y_batch is None:
            self.y_batch = pydtnn.backends.gpu.tensor_gpu.TensorGPU(
                gpuarray.empty((self.batch_size, *self.layers[-1].shape), self.dtype),
                self.tensor_format, self.cudnn_dtype)

        self.history = {lm: [] for lm in (self.loss_and_metrics + [f"val_{m}" for m in self.loss_and_metrics])}

        terminate = False

        for epoch in range(self.num_epochs):

            train_batch_generator, val_batch_generator = self.dataset.get_train_val_generator()

            train_total_loss, train_batch_count = np.zeros(len(self.loss_and_metrics)), 0
            val_total_loss, val_batch_count = np.zeros(len(self.loss_and_metrics)), 0

            if self.rank == 0:
                fmt = "%%%dd" % (len(str(self.num_epochs)))
                epoch_string = "Epoch %s/%s" % (fmt, fmt)
                pbar = tqdm(total=self.dataset.train_nsamples, ncols=bar_width,
                            ascii=" ▁▂▃▄▅▆▇█", smoothing=0.3,
                            desc=epoch_string % (epoch + 1, self.num_epochs), unit=" samples")

            for lr_sched in self.lr_schedulers:
                lr_sched.on_epoch_begin(self, self.rank)

            for x_batch, y_batch, batch_size in train_batch_generator:
                tic = timer()
                train_batch_loss = self._train_batch(x_batch, y_batch, batch_size)
                toc = timer()
                train_total_loss, train_batch_count, string = \
                    self._update_running_average(train_batch_loss, train_total_loss,
                                                 train_batch_count, batch_size)
                if self.rank == 0:
                    # noinspection PyUnboundLocalVariable
                    pbar.set_postfix_str(s=string, refresh=True)
                    pbar.update(batch_size)
                    self.perf_counter.add_training_time_and_batch_size(epoch, toc - tic, batch_size)

            if self.rank == 0:
                pbar.close()
                for c in range(len(self.loss_and_metrics)):
                    self.history[self.loss_and_metrics[c]].append(train_total_loss[c])

            for x_batch, y_batch, batch_size in val_batch_generator:
                val_batch_loss = self._evaluate_batch(x_batch, y_batch, batch_size)
                val_total_loss, val_batch_count, string = \
                    self._update_running_average(val_batch_loss, val_total_loss,
                                                 val_batch_count, batch_size, prefix="val_")
                if self.rank == 0:
                    print("\033[A\033[%dC\b, %s]" % (bar_width, string))

            if self.rank == 0:
                for c in range(len(self.loss_and_metrics)):
                    self.history["val_" + self.loss_and_metrics[c]].append(val_total_loss[c])

            for lr_sched in self.lr_schedulers:
                lr_sched.on_epoch_end(train_total_loss, val_total_loss)
                if getattr(lr_sched, "stop_training", False):
                    terminate = True

            if terminate:
                break

        self.tracer.define_event_types(self)
        return self.history

    def _evaluate_batch(self, x_batch, y_batch, current_batch_size):
        self.mode = EVALUATE_MODE

        try:
            x, y_targ = self._get_x_y_targ(x_batch, y_batch, current_batch_size)
        except ValueError:
            return self.total_metrics

        # Forward pass (FP)
        for i in range(1, len(self.layers)):
            self.tracer.emit_event(PYDTNN_MDL_EVENT, self.layers[i].id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_FORWARD)
            x = self.layers[i].forward(x)
            self.tracer.emit_event(PYDTNN_MDL_EVENT, 0)

        y_pred = self.layers[-1].y
        loss, _ = self.loss_func(y_pred, y_targ, self.batch_size)
        self.total_metrics, _ = self._compute_metrics_funcs(y_pred, y_targ, loss)
        return self.total_metrics

    def evaluate(self, x_test, y_test, bar_width=110):
        self.dataset = CustomDataset(self, x_test=x_test, y_test=y_test)
        self.evaluate_dataset(bar_width=bar_width)

    @ensure_model_is_initialized
    def evaluate_dataset(self, bar_width=120):
        if self.enable_cudnn and self.y_batch is None:
            self.y_batch = pydtnn.backends.gpu.tensor_gpu.TensorGPU(
                gpuarray.empty((self.batch_size, *self.layers[-1].shape), self.dtype),
                self.tensor_format, self.cudnn_dtype)

        test_batch_generator = self.dataset.get_test_generator()

        if self.rank == 0:
            test_total_loss, test_batch_count = np.zeros(len(self.loss_and_metrics)), 0
            pbar = tqdm(total=self.dataset.test_nsamples, ncols=bar_width,
                        ascii=" ▁▂▃▄▅▆▇█", smoothing=0.3,
                        desc="Testing", unit=" samples")

        for x_batch, y_batch, batch_size in test_batch_generator:
            tic = timer()
            test_batch_loss = self._evaluate_batch(x_batch, y_batch, batch_size)
            toc = timer()
            if self.rank == 0:
                # noinspection PyUnboundLocalVariable
                test_total_loss, test_batch_count, string = \
                    self._update_running_average(test_batch_loss, test_total_loss, test_batch_count, batch_size,
                                                 prefix="test_")
                # noinspection PyUnboundLocalVariable
                pbar.set_postfix_str(s=string, refresh=True)
                pbar.update(batch_size)
                self.perf_counter.add_testing_time_and_batch_size(self._evaluate_round, toc - tic, batch_size)

        # Increment self._evaluate_round
        self._evaluate_round += 1

        if self.rank == 0:
            pbar.close()
            # Sleep for half a second to allow pbar to write its output before returning
            time.sleep(.5)
