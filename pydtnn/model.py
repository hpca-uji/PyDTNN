"""
PyDTNN model
"""

#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021 Universitat Jaume I
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

import importlib
import resource
import sys
import time
from collections import defaultdict
from timeit import default_timer as timer

from tqdm import tqdm

import pydtnn.metrics
from . import optimizers, losses, metrics
from . import utils
from .datasets.dataset import Dataset
from .parser import parser
from .performance_models import *
from .tracers import PYDTNN_MDL_EVENT, PYDTNN_MDL_EVENTS, PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, ExtraeTracer, \
    SimpleTracer, PYDTNN_MDL_UPDATE_DW, PYDTNN_OPS_ALLREDUCE_DW, PYDTNN_MDL_WAIT_DW, \
    PYDTNN_MDL_FORWARD, PYDTNN_MDL_BACKWARD, PYDTNN_MDL_ALLREDUCE_DW

supported_gpu = False
supported_cudnn = True
supported_nccl = True
supported_mpi4py = True
enable_cudnn = False


try:
    from mpi4py import MPI
except (ImportError, ModuleNotFoundError):
    supported_mpi4py = False

EVALUATE_MODE, TRAIN_MODE = (0, 1)


class PerformanceCounter:
    TRAINING, TESTING = range(2)

    def __init__(self):
        self._in_second_testing_round = False
        self._times_record = defaultdict(lambda: defaultdict(lambda: []))
        self._batch_sizes_record = defaultdict(lambda: defaultdict(lambda: []))
        self._memory_record = defaultdict(lambda: defaultdict(lambda: []))

    # -------------------------------
    #  Public methods and properties
    # -------------------------------

    def add_training_time_and_batch_size(self, epoch, elapsed_time, batch_size):
        self._add_time_and_batch_size(self.TRAINING, epoch, elapsed_time, batch_size)

    def add_testing_time_and_batch_size(self, elapsed_time, batch_size):
        self._add_time_and_batch_size(self.TESTING, 0, elapsed_time, batch_size)

    def init_testing_data(self):
        """
        Should be called before a testing round to perform the next actions:
          * Keep the memory consumption of the first testing round.
          * Clear the time and size data of the first testing round.
        """
        if len(self._times_record[self.TESTING]):
            self._in_second_testing_round = True
            self._times_record[self.TESTING] = defaultdict(lambda: [])
            self._batch_sizes_record[self.TESTING] = defaultdict(lambda: [])

    @property
    def training_throughput(self):
        return self._throughput(self.TRAINING)

    @property
    def training_throughput_only_last_half_of_each_epoch(self):
        return self._throughput(self.TRAINING, last_half=True)

    @property
    def num_epochs(self):
        return len(self._batch_sizes_record[self.TRAINING].keys())

    @property
    def training_time(self):
        return self._time(self.TRAINING)

    @property
    def training_time_estimated_from_last_half_of_each_epoch(self):
        return self._time(self.TRAINING, last_half=True)

    @property
    def training_maximum_memory(self):
        return self._maximum_memory(self.TRAINING)

    @property
    def training_mean_memory(self):
        return self._mean_memory(self.TRAINING)

    @property
    def testing_throughput(self):
        return self._throughput(self.TESTING)

    @property
    def testing_time(self):
        return self._time(self.TESTING)

    @property
    def testing_maximum_memory(self):
        return self._maximum_memory(self.TESTING)

    @property
    def testing_mean_memory(self):
        return self._mean_memory(self.TESTING)

    # -------------------------------
    #  Private methods
    # -------------------------------

    def _add_time_and_batch_size(self, where, epoch, elapsed_time, batch_size):
        self._times_record[where][epoch].append(elapsed_time)
        self._batch_sizes_record[where][epoch].append(batch_size)
        if where == self.TESTING and self._in_second_testing_round:
            return
        mem = (resource.getrusage(resource.RUSAGE_SELF)[2]
               + resource.getrusage(resource.RUSAGE_CHILDREN)[2])
        self._memory_record[where][epoch].append(mem)  # KiB in GNU/Linux

    def _time(self, where, last_half=False):
        # When last_half is True, the total time is estimated from the last half steps of each epoch time
        if not last_half:
            times_per_epoch = [np.sum(t_array) for t_array in self._times_record[where].values()]
        else:
            times_per_epoch = []
            for t_array in self._times_record[where].values():
                t_array_last_half = t_array[len(t_array) // 2:]
                if len(t_array_last_half) > 0:
                    times_per_epoch.append(np.sum(t_array_last_half) * len(t_array) / len(t_array_last_half))
        return np.sum(times_per_epoch)

    def _size(self, where, last_half=False):
        # When last_half is True, the total size is estimated from the last half steps of each epoch size
        if not last_half:
            batch_sizes_per_epoch = [np.sum(s_array) for s_array in self._batch_sizes_record[where].values()]
        else:
            batch_sizes_per_epoch = []
            for s_array in self._batch_sizes_record[where].values():
                s_array_last_half = s_array[len(s_array) // 2:]
                if len(s_array_last_half) > 0:
                    batch_sizes_per_epoch.append(np.sum(s_array_last_half) * len(s_array) / len(s_array_last_half))
        return np.sum(batch_sizes_per_epoch)

    def _throughput(self, where, last_half=False):
        return self._size(where, last_half) / self._time(where, last_half)

    def _maximum_memory(self, where):
        maximum_memory_per_epoch = [np.max(m_array) for m_array in self._memory_record[where].values()]
        return np.max(maximum_memory_per_epoch)

    def _mean_memory(self, where):
        mean_memory_per_epoch = [np.mean(m_array[len(m_array) // 2:])
                                 for m_array in self._memory_record[where].values()]
        return np.mean(mean_memory_per_epoch)


def _layer_id_generator():
    """To obtain consecutive layer ids. See Layer.set_model()."""
    current_layer_id = 0
    while True:
        yield current_layer_id
        current_layer_id += 1


class Model:
    """
    PyDTNN Model
    """

    def __init__(self, comm=None, non_blocking_mpi=False, enable_gpu=False, enable_gpudirect=False,
                 enable_nccl=False, dtype=np.float32, tracing=False, tracer_output="", **kwargs):
        # Attributes related to the given arguments
        self.comm = comm
        self.blocking_mpi = not non_blocking_mpi
        global enable_cudnn
        enable_cudnn = self.enable_cudnn = enable_gpu
        self.gpudirect = enable_gpudirect
        self.enable_nccl = enable_nccl
        self.dtype = dtype
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
        # Layers attributes
        self.layers = []
        self.layer_id = _layer_id_generator()
        # In data parallel, we assume that file weights are stored in a nfs mounted directory.
        self.shared_storage = True
        # Matmul
        self.matmul = getattr(utils, "matmul")
        # Execution attributes
        self.nparams = 0
        self.rank = 0
        self.nprocs = 1
        self.mode = TRAIN_MODE
        if self.comm and supported_mpi4py:
            self.rank = self.comm.Get_rank()
            self.nprocs = self.comm.Get_size()
        elif self.comm:
            print("Please, install mpi4py to allow parallel MPI execution!")
            sys.exit(-1)
        if self.enable_cudnn:

            supported_cudnn = True
            supported_nccl = True
            try:
                import pydtnn.backends.gpu.tensor_gpu
                global gpuarray
                import pycuda.gpuarray as gpuarray
                import pycuda.driver as drv
                from pydtnn.backends.gpu.libs import libcudnn as cudnn
                # noinspection PyUnresolvedReferences
                from skcuda import cublas
            except (ImportError, ModuleNotFoundError, OSError):
                supported_cudnn = False
            try:
                from pydtnn.backends.gpu.libs import libnccl as nccl
            except (ImportError, ModuleNotFoundError, OSError):
                supported_nccl = False

            if not supported_cudnn:
                print("Please, install pycuda, skcuda and cudnn to be able to use the GPUs!")
                sys.exit(-1)

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

            if self.enable_nccl and self.comm:
                if not supported_nccl:
                    print("Please, install nccl to be able to use NVIDIA NCCL inter-GPU communications!")
                    sys.exit(-1)

                types = {np.float64: nccl.DataType.Float64,
                         np.float32: nccl.DataType.Float32,
                         np.int8: nccl.DataType.Int8,
                         np.int32: nccl.DataType.Int32}

                self.nccl_type = types.get(self.dtype, nccl.DataType.Float32)

                hostname = MPI.Get_processor_name()

                hosts_data = comm.allgather([self.rank, hostname])
                # Build a dictionary hostname : [ranks_in_host]
                #   { "host1": [0, 1], "host2": [2, 3] }
                hosts = {}
                for r, h in hosts_data:
                    hosts.setdefault(h, []).append(r)

                # Check that no more processes than GPUs per node are used
                self.gpus_per_node = self.kwargs.get('gpus_per_node', 1)
                for host, ranks_in_host in hosts.items():
                    assert len(ranks_in_host) <= self.gpus_per_node

                nccl_id = comm.bcast(nccl.ncclGetUniqueId() if self.rank == 0 else None)
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
                #     # Get an id once per master process and distribute it to all intra ranks
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
        # Read model
        self.model_name = self.kwargs.get("model_name")
        if self.model_name:
            self.read_model(self.model_name)
        # Explicit declaration of those model attributes that are referenced by other parts of PyDTNN
        self.batch_size = self.kwargs['batch_size']
        self.steps_per_epoch = self.kwargs['steps_per_epoch']
        self.cpu_speed = self.kwargs['cpu_speed']
        self.memory_bw = self.kwargs['memory_bw']
        self.network_bw = self.kwargs['network_bw']
        self.network_lat = self.kwargs['network_lat']
        self.network_alg = self.kwargs['network_alg']
        # Attributes that will be properly defined elsewhere
        self.y_batch = None
        self.history = None

    def __getattr__(self, item):
        try:
            return self.kwargs[item]
        except KeyError:
            raise AttributeError(f"'Model' object has no attribute '{item}'") from None

    def read_model(self, model_name):
        try:
            model_module = importlib.import_module(f"pydtnn.models.{model_name}")
            getattr(model_module, f"create_{model_name}")(self)
        except (ModuleNotFoundError, AttributeError):
            import traceback
            print(traceback.format_exc())
            sys.exit(-1)

    def show(self):
        bfp = {np.float32: 4, np.float64: 8}[self.dtype]
        line = "+-------+--------------------------+---------+---------------+-------------------" \
               "+------------------------+"
        head = "| Layer |           Type           | #Params | Output shape  |   Weights shape   " \
               "|       Parameters       |"
        print(line)
        print(head)
        for layer in self.layers:
            print(line)
            layer.show()
        print(line)
        print(f"|{'':^7s} {'Total parameters':^26s} {self.nparams:^9d} {utils.convert_size(self.nparams * bfp):^15s} "
              f"{'':19s} {'':24s}|")
        print(line)

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

    def __apply_relu_fusion(self):
        """ Apply Relu fusion in a recursive manner """

        def __relu_fusion(layers):
            fused_layers = []
            for i, curr_layer in enumerate(layers):
                if curr_layer.is_block_layer:
                    for j, p in enumerate(curr_layer.paths):
                        curr_layer.paths[j] = __relu_fusion(p)
                elif i > 0 and type(curr_layer).__name__ == "Relu" and \
                        type(fused_layers[-1]).__name__ in ["Conv2D", "BatchNormalization"]:
                    prev_layer = fused_layers.pop()
                    print("Fusing %03d_%s with %03d_%s ..." % (prev_layer.id, type(prev_layer).__name__,
                                                               curr_layer.id, type(curr_layer).__name__))
                    curr_layer = getattr(importlib.import_module("pydtnn.layers"),
                                         type(prev_layer).__name__ + type(curr_layer).__name__)(from_parent=prev_layer)
                fused_layers.append(curr_layer)
            return fused_layers

        if not self.enable_cudnn:
            self.layers = __relu_fusion(self.layers)

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

    def __compute_metrics_funcs(self, y_pred, y_targ, loss, metrics_funcs, blocking=True):
        loss_req = None
        if self.enable_cudnn:
            _losses = np.array([loss] + [func(y_pred.ary, y_targ.ary)
                                         for func in metrics_funcs], dtype=np.float32) / self.nprocs
        else:
            _losses = np.array([loss] + [func(y_pred, y_targ) for func in metrics_funcs],
                               dtype=np.float32) / self.nprocs
        if self.comm is not None and blocking:
            self.comm.Allreduce(MPI.IN_PLACE, _losses, op=MPI.SUM)
        elif self.comm is not None and not blocking:
            loss_req = self.comm.Iallreduce(MPI.IN_PLACE, _losses, op=MPI.SUM)
        return _losses, loss_req

    @staticmethod
    def __update_running_average(curr, total, count, batch_size, loss_metrics, prefix=""):
        string = ""
        total = ((curr * batch_size) + (total * count)) / (count + batch_size)
        for c in range(len(loss_metrics)):
            loss_str = pydtnn.metrics.metric_format.get(loss_metrics[c], loss_metrics[c])
            string += ("%s, " % (prefix + loss_str)) % total[c]
        string = string[:-2]
        return total, count + batch_size, string

    def __train_batch(self, x_batch, y_batch, local_batch_size, global_batch_size,
                      loss_func, metrics_funcs, optimizer, lr_schedulers):

        self.mode = TRAIN_MODE
        for lr_sched in lr_schedulers:
            lr_sched.on_batch_begin(self, optimizer, self.rank)

        if self.enable_cudnn:
            if x_batch.shape[0] != local_batch_size:
                return self.total_metrics
            self.layers[0].y.ary.set(x_batch)
            self.y_batch.ary.set(y_batch)
            x, y_targ = self.layers[0].y, self.y_batch
        else:
            x, y_targ = x_batch, y_batch

        # Forward pass (FP)
        for i in range(1, len(self.layers)):
            self.tracer.emit_event(PYDTNN_MDL_EVENT, self.layers[i].id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_FORWARD)
            x = self.layers[i].forward(x)
            self.tracer.emit_event(PYDTNN_MDL_EVENT, 0)

        loss, dx = loss_func(x, y_targ, global_batch_size)
        self.total_metrics, _ = self.__compute_metrics_funcs(x, y_targ, loss, metrics_funcs)

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
                self.layers[i].update_weights(optimizer)
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
                self.layers[i].update_weights(optimizer)
                self.tracer.emit_event(PYDTNN_MDL_EVENT, 0)

        if self.enable_cudnn:
            for i in range(len(self.layers) - 1, 0, -1):
                if self.layers[i].grad_vars:
                    self.layers[i].stream_2.synchronize()

        for lr_sched in lr_schedulers:
            lr_sched.on_batch_end(self, optimizer, self.rank)

        return self.total_metrics

    def train(self, x_train, y_train, x_val, y_val, nepochs, local_batch_size,
              loss="categorical_cross_entropy", metrics_list=("categorical_accuracy",),
              optimizer=optimizers.SGD(), bar_width=110):

        dataset = Dataset(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
        history = self.train_dataset(dataset, nepochs, local_batch_size, 0,
                                     loss=loss, metrics_list=metrics_list, optimizer=optimizer,
                                     bar_width=bar_width)
        return history

    def train_dataset(self, dataset, nepochs, local_batch_size, val_split=0.2,
                      loss="categorical_cross_entropy", metrics_list=("categorical_accuracy",),
                      optimizer=optimizers.SGD(), lr_schedulers=(), bar_width=110):
        if self.enable_cudnn and self.y_batch is None:
            self.y_batch = pydtnn.backends.gpu.tensor_gpu.TensorGPU(
                gpuarray.empty((local_batch_size, *self.layers[-1].shape), self.dtype),
                self.tensor_format, self.cudnn_dtype)
        loss_func = getattr(losses, loss)(shape=(local_batch_size, *self.layers[-1].shape), model=self)
        metrics_funcs = [getattr(metrics, m)(shape=(local_batch_size, *self.layers[-1].shape), model=self) for m in
                         metrics_list]
        loss_metrics = [loss] + metrics_list
        self.history = {lm: [] for lm in (loss_metrics + [f"val_{m}" for m in loss_metrics])}
        if self.enable_cudnn:
            # noinspection PyUnresolvedReferences
            optimizer.set_gpudirect(self.gpudirect)

        dataset.make_train_val_partitions(val_split)
        self.steps_per_epoch = dataset.train_nsamples / (local_batch_size * self.nprocs)
        terminate = False

        for epoch in range(nepochs):

            train_batch_generator, val_batch_generator = \
                dataset.get_train_val_generator(local_batch_size, self.rank, self.nprocs, val_split)

            train_total_loss, train_batch_count = np.zeros(len(loss_metrics)), 0
            val_total_loss, val_batch_count = np.zeros(len(loss_metrics)), 0

            if self.rank == 0:
                fmt = "%%%dd" % (len(str(nepochs)))
                epoch_string = "Epoch %s/%s" % (fmt, fmt)
                pbar = tqdm(total=dataset.train_nsamples, ncols=bar_width,
                            ascii=" ▁▂▃▄▅▆▇█", smoothing=0.3,
                            desc=epoch_string % (epoch + 1, nepochs), unit=" samples")

            for lr_sched in lr_schedulers:
                lr_sched.on_epoch_begin(self, self.rank)

            for x_batch, y_batch, batch_size in train_batch_generator:
                tic = timer()
                train_batch_loss = self.__train_batch(x_batch, y_batch, local_batch_size, batch_size,
                                                      loss_func, metrics_funcs, optimizer, lr_schedulers)
                toc = timer()
                train_total_loss, train_batch_count, string = \
                    self.__update_running_average(train_batch_loss, train_total_loss,
                                                  train_batch_count, batch_size,
                                                  loss_metrics)
                if self.rank == 0:
                    # noinspection PyUnboundLocalVariable
                    pbar.set_postfix_str(s=string, refresh=True)
                    pbar.update(batch_size)
                    self.perf_counter.add_training_time_and_batch_size(epoch, toc - tic, batch_size)

            if self.rank == 0:
                pbar.close()
                for c in range(len(loss_metrics)):
                    self.history[loss_metrics[c]].append(train_total_loss[c])

            for x_batch, y_batch, batch_size in val_batch_generator:
                val_batch_loss = self.__evaluate_batch(x_batch, y_batch, local_batch_size, batch_size,
                                                       loss_func, metrics_funcs)
                val_total_loss, val_batch_count, string = \
                    self.__update_running_average(val_batch_loss, val_total_loss,
                                                  val_batch_count, batch_size,
                                                  loss_metrics, prefix="val_")
                if self.rank == 0:
                    print("\033[A\033[%dC\b, %s]" % (bar_width, string))

            if self.rank == 0:
                for c in range(len(loss_metrics)):
                    self.history["val_" + loss_metrics[c]].append(val_total_loss[c])

            for lr_sched in lr_schedulers:
                lr_sched.on_epoch_end(self, optimizer, loss_metrics,
                                      train_total_loss, val_total_loss, self.rank)
                if getattr(lr_sched, "stop_training", False):
                    terminate = True

            if terminate:
                break

        self.tracer.define_event_types(self)
        return self.history

    def __evaluate_batch(self, x_batch, y_batch, local_batch_size, global_batch_size, loss_func, metrics_funcs):
        self.mode = EVALUATE_MODE

        if self.enable_cudnn:
            if x_batch.shape[0] != local_batch_size:
                return self.total_metrics
            self.layers[0].y.ary.set(x_batch)
            self.y_batch.ary.set(y_batch)
            x, y_targ = self.layers[0].y, self.y_batch
        else:
            x, y_targ = x_batch, y_batch

        # Forward pass (FP)
        for i in range(1, len(self.layers)):
            self.tracer.emit_event(PYDTNN_MDL_EVENT, self.layers[i].id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_FORWARD)
            x = self.layers[i].forward(x)
            self.tracer.emit_event(PYDTNN_MDL_EVENT, 0)

        y_pred = self.layers[-1].y
        loss, _ = loss_func(y_pred, y_targ, global_batch_size)
        self.total_metrics, _ = self.__compute_metrics_funcs(y_pred, y_targ, loss, metrics_funcs)
        return self.total_metrics

    def evaluate(self, x_test, y_test, local_batch_size,
                 loss="categorical_cross_entropy", metrics_list=("categorical_accuracy",),
                 bar_width=110):

        dataset = Dataset(x_test=x_test, y_test=y_test)
        self.evaluate_dataset(dataset, local_batch_size, loss=loss, metrics_list=metrics_list, bar_width=bar_width)

    def evaluate_dataset(self, dataset, local_batch_size,
                         loss="categorical_cross_entropy", metrics_list=("categorical_accuracy",),
                         bar_width=120):
        if self.enable_cudnn and self.y_batch is None:
            self.y_batch = pydtnn.backends.gpu.tensor_gpu.TensorGPU(
                gpuarray.empty((local_batch_size, *self.layers[-1].shape), self.dtype),
                self.tensor_format, self.cudnn_dtype)
        loss_func = getattr(losses, loss)(shape=(self.batch_size, *self.layers[-1].shape), model=self)
        metrics_funcs = [getattr(metrics, m)(shape=(self.batch_size, *self.layers[-1].shape), model=self) for m in
                         metrics_list]
        loss_metrics = [loss] + metrics_list
        test_batch_generator = dataset.get_test_generator(local_batch_size, self.rank, self.nprocs)

        if self.kwargs.get("enable_fused_relus"):
            self.__apply_relu_fusion()

        if self.rank == 0:
            test_total_loss, test_batch_count = np.zeros(len(loss_metrics)), 0
            pbar = tqdm(total=dataset.test_nsamples, ncols=bar_width,
                        ascii=" ▁▂▃▄▅▆▇█", smoothing=0.3,
                        desc="Testing", unit=" samples")

        self.perf_counter.init_testing_data()  # Only the last testing data should be kept

        for x_batch, y_batch, batch_size in test_batch_generator:
            tic = timer()
            test_batch_loss = self.__evaluate_batch(x_batch, y_batch, self.batch_size, batch_size,
                                                    loss_func, metrics_funcs)
            toc = timer()
            if self.rank == 0:
                # noinspection PyUnboundLocalVariable
                val_total_loss, val_batch_count, string = \
                    self.__update_running_average(test_batch_loss, test_total_loss, test_batch_count, batch_size,
                                                  loss_metrics, prefix="test_")
                # noinspection PyUnboundLocalVariable
                pbar.set_postfix_str(s=string, refresh=True)
                pbar.update(batch_size)
                self.perf_counter.add_testing_time_and_batch_size(toc - tic, batch_size)

        if self.rank == 0:
            pbar.close()
            # Sleep for half a second to allow pbar to write its output before returning
            time.sleep(.5)

        self.tracer.define_event_types(self)
