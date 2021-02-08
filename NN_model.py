"""
Model definition for Python Distributed Training of Neural Networks (PyDTNN)

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

Copyright 2021 Universitat Jaume I

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

__author__ = "Manuel F. Dolz, Enrique S. Quintana, Sergio Barrachina, Mar Catalán, Adrián Castelló"
__contact__ = "dolzm@uji.es"
__copyright__ = "Copyright 2021, Universitat Jaume I"
__credits__ = ["Manuel F. Dolz, Enrique S. Quintana", "Sergio Barrachina", "Mar Catalán", "Adrián Castelló"]
__date__ = "2020/03/22"

__email__ = "dolzm@uji.es"
__license__ = "GPLv3"
__maintainer__ = "Manuel F. Dolz"
__status__ = "Production"
__version__ = "1.1.0"

import resource
import sys
import time
from collections import defaultdict

from tqdm import tqdm
from timeit import default_timer as timer

import NN_optimizer
import NN_util
import datasets.NN_dataset
from NN_sim import *
from NN_tracer import PYDTNN_MDL_EVENT, PYDTNN_MDL_EVENTS, \
    PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, ExtraeTracer, SimpleTracer, PYDTNN_MDL_UPDATE_DW, PYDTNN_OPS_ALLREDUCE_DW, \
    PYDTNN_MDL_WAIT_DW, PYDTNN_MDL_FORWARD, PYDTNN_MDL_BACKWARD, PYDTNN_MDL_ALLREDUCE_DW

supported_gpu = False
supported_cudnn = True
supported_mpi4py = True
enable_cudnn = False
try:
    import pycuda.gpuarray as gpuarray
    import pycuda.driver as drv
    import libcudnn.libcudnn as cudnn
    import libnccl.libnccl as nccl
    from skcuda import cublas
except Exception as e:
    supported_cudnn = False
    print(e)

try:
    from mpi4py import MPI
except Exception as e:
    supported_mpi4py = False
    print(e)


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

    def add_training_time_and_batch_size(self, epoch, time, batch_size):
        self._add_time_and_batch_size(self.TRAINING, epoch, time, batch_size)

    def add_testing_time_and_batch_size(self, time, batch_size):
        self._add_time_and_batch_size(self.TESTING, 0, time, batch_size)

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

    def _add_time_and_batch_size(self, where, epoch, time, batch_size):
        self._times_record[where][epoch].append(time)
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
            times_per_epoch = [np.sum(t_array[len(t_array) // 2:]) * len(t_array) / (len(t_array) // 2)
                               for t_array in self._times_record[where].values()]
        return np.sum(times_per_epoch)

    def _size(self, where, last_half=False):
        # When last_half is True, the total size is estimated from the last half steps of each epoch size
        if not last_half:
            batch_sizes_per_epoch = [np.sum(s_array) for s_array in self._batch_sizes_record[where].values()]
        else:
            batch_sizes_per_epoch = [np.sum(s_array[len(s_array) // 2:]) * len(s_array) / (len(s_array) // 2)
                                     for s_array in self._batch_sizes_record[where].values()]
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


class Model:

    def __init__(self, params, comm=None, non_blocking_mpi=False,
                 enable_gpu=False, enable_gpudirect=False, enable_nccl=False, dtype=np.float32,
                 tracing=False, simple_tracer_output=""):
        self.id = 0
        self.layers = []
        self.params = params
        self.comm = comm
        self.blocking_mpi = not non_blocking_mpi
        if simple_tracer_output == "":
            self.tracer = ExtraeTracer(tracing)
        else:
            self.tracer = SimpleTracer(tracing, simple_tracer_output, self.comm)
        self.perf_counter = PerformanceCounter()
        global enable_cudnn
        enable_cudnn = self.enable_cudnn = enable_gpu
        self.gpudirect = enable_gpudirect
        self.enable_nccl = enable_nccl
        self.dtype = dtype
        # In data parallel, we assume that file weights are stored in a nfs mounted directory.
        self.params.shared_storage = True

        self.nparams = 0
        self.rank = 0
        self.nprocs = 1

        if self.comm and supported_mpi4py:
            self.rank = self.comm.Get_rank()
            self.nprocs = self.comm.Get_size()
        elif self.comm:
            print("You must install mpi4py to allow parallel MPI execution!")
            sys.exit(-1)

        if self.enable_cudnn:
            if not supported_cudnn:
                print("You must install pycuda, skcuda, cudnn, and, optionally, nccl, to be able to use the GPUs!")
                sys.exit(-1)
            import pycuda.autoinit
            # # Uncomment the next code if pycuda.autoinit is not available
            # device_id = self.rank % drv.Device.count()
            # drv.init()
            # context = drv.Device(device_id).make_context()
            # import atexit
            # atexit.register(context.pop)

            if self.enable_nccl and self.comm:
                types = {np.float64: nccl.DataType.Float64,
                         np.float32: nccl.DataType.Float32,
                         np.int8: nccl.DataType.Int8,
                         np.int32: nccl.DataType.Int32}

                self.nccl_type = types.get(self.type, nccl.DataType.Float32)

                hostname = MPI.Get_processor_name()

                hosts_data = comm.allgather([self.rank, hostname])
                # Build a dictionary hostname : [ranks_in_host]
                #   { "host1": [0, 1], "host2": [2, 3] }
                hosts = {}
                for r, h in hosts_data:
                    hosts.setdefault(h, []).append(r)

                # Check that no more processes than GPUs per node are used
                for host, ranks_in_host in hosts.items():
                    assert len(ranks_in_host) <= self.params.gpus_per_node

                id = comm.bcast(nccl.ncclGetUniqueId() if self.rank == 0 else None)
                self.nccl_comm = nccl.ncclCommInitRank(self.nprocs, id, self.rank)

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

            cudnn_type = types.get(self.type, "CUDNN_DATA_FLOAT")

            self.cudnn_dtype = cudnn.cudnnDataType[cudnn_type]
            self.tensor_fmt = cudnn.cudnnTensorFormat['CUDNN_TENSOR_NCHW']

    def show(self):
        bfp = {np.float32: 4, np.float64: 8}[self.dtype]
        print(
            "+-------+--------------------------+---------+---------------+-------------------+------------------------+")
        print(
            "| Layer |           Type           | #Params | Output shape  |   Weights shape   |       Parameters       |")
        for l in self.layers:
            print(
                '+-------+--------------------------+---------+---------------+-------------------+------------------------+')
            l.show()
        print(
            '+-------+--------------------------+---------+---------------+-------------------+------------------------+')
        print(
            f"|{'':^7s} {'Total parameters':^26s} {self.nparams:^9d} {NN_util.convert_size(self.nparams * bfp):^15s} {'':19s} {'':24s}|")
        print(
            '+-------+--------------------------+---------+---------------+-------------------+------------------------+')

    def add(self, layer):
        layer.id = self.id
        layer.tracer = self.tracer
        layer.dtype = self.dtype
        layer.model = self
        layer.batch_size = self.params.batch_size

        need_dx = layer.id > 1
        prev_shape = self.layers[-1].shape if layer.id > 0 else ()

        layer.gpudirect = self.gpudirect

        if self.enable_cudnn:
            layer.cublas_handle = self.cublas_handle
            layer.cudnn_handle = self.cudnn_handle
            layer.stream = self.stream
            layer.cudnn_dtype = self.cudnn_dtype
            layer.tensor_fmt = self.tensor_fmt
            y = self.layers[-1].y if layer.id > 0 else None
            layer.initialize(prev_shape, need_dx, y)
        else:
            layer.matmul = getattr(NN_util, "matmul")
            layer.initialize(prev_shape, need_dx)

        self.nparams += layer.nparams
        self.layers.append(layer)
        self.id += 1

        if layer.act:
            self.add(layer.act())

    def load_weights_and_bias(self, filename):
        d = np.load(filename)
        for l, layer in enumerate(self.layers):
            base = "%s_%s" % (str(l), type(layer).__name__)
            for p in layer.grad_vars:
                key = "%s_%s" % (base, p)
                if key in d.files:
                    if self.enable_cudnn:
                        getattr(layer, p).ary.set(d[key])
                    else:
                        setattr(layer, p, d[key])
                else:
                    print("Could not find %s for layer %s in %s file!" % (p, base, filename))

    def store_weights_and_bias(self, filename):
        if self.params.shared_storage and self.rank == 0:
            d = {}
            for l, layer in enumerate(self.layers):
                base = "%s_%s" % (str(l), type(layer).__name__)
                for p in layer.grad_vars:
                    key = "%s_%s" % (base, p)
                    d[key] = getattr(layer, p)
                    if self.enable_cudnn:
                        d[key] = d[key].ary.get()
            np.savez_compressed(filename, **d)

    def calculate_time(self):
        total_time = np.zeros((4,), dtype=np.float32)  # Total time, Comp time, Memo time, Net time

        # Forward pass (FP)
        for l in range(1, len(self.layers)):
            total_time += self.layers[l].fwd_time

        if self.blocking_mpi:
            # Blocking MPI
            # Back propagation. Gradient computation (GC) and weights update (WU)
            for l in range(len(self.layers) - 1, 0, -1):
                total_time += self.layers[l].bwd_time

            # Weight update (WU)
            for l in range(len(self.layers) - 1, 0, -1):
                if self.comm and self.layers[l].weights.size > 0:
                    total_time += allreduce_time(self.layers[l].weights.size + self.layers[l].biases.size,
                                                 self.params.cpu_speed, self.params.network_bw, self.params.network_lat,
                                                 self.params.network_alg, self.nprocs, self.dtype)
        else:
            total_time_iar = 0
            # Non-blocking MPI
            # Back propagation. Gradient computation (GC) and weights update (WU)
            for l in range(len(self.layers) - 1, 0, -1):
                total_time += self.layers[l].bwd_time
                if self.comm and self.layers[l].weights.size > 0:
                    time_iar = allreduce_time(self.layers[l].weights.size + self.layers[l].biases.size,
                                              self.params.cpu_speed, self.params.network_bw, self.params.network_lat,
                                              self.params.network_alg, self.nprocs, self.dtype)
                    total_time[3] += time_iar[3]
                    total_time_iar = max(total_time[0], total_time_iar) + time_iar[0]

            total_time[0] = max(total_time[0], total_time_iar)

        return total_time

    def __compute_metrics_funcs(self, Y_pred, Y_targ, loss, metrics_funcs, blocking=True):
        loss_req = None
        if self.enable_cudnn:
            losses = np.array([loss] + [func(Y_pred.ary, Y_targ.ary)
                                        for func in metrics_funcs], dtype=np.float32) / self.nprocs
        else:
            losses = np.array([loss] + [func(Y_pred, Y_targ) for func in metrics_funcs], dtype=np.float32) / self.nprocs
        if self.comm is not None and blocking:
            self.comm.Allreduce(MPI.IN_PLACE, losses, op=MPI.SUM)
        elif self.comm is not None and not blocking:
            loss_req = self.comm.Iallreduce(MPI.IN_PLACE, losses, op=MPI.SUM)
        return losses, loss_req

    @staticmethod
    def __update_running_average(curr, total, count, batch_size, loss_metrics, prefix=""):
        string = ""
        total = ((curr * batch_size) + (total * count)) / (count + batch_size)
        for c in range(len(loss_metrics)):
            loss_str = NN_util.metric_format.get(loss_metrics[c], loss_metrics[c])
            string += ("%s, " % (prefix + loss_str)) % total[c]
        string = string[:-2]
        return total, count + batch_size, string

    def __train_batch(self, X_batch, Y_batch, local_batch_size, global_batch_size,
                      loss_func, metrics_funcs, optimizer, lr_schedulers):

        self.mode = "train"
        for lr_sched in lr_schedulers:
            lr_sched.on_batch_begin(self, optimizer, self.rank)

        if self.enable_cudnn:
            if X_batch.shape[0] != local_batch_size:
                return self.total_metrics
            self.layers[0].y.ary.set(X_batch)
            self.Y_batch.ary.set(Y_batch)
            x, Y_targ = self.layers[0].y, self.Y_batch
        else:
            x, Y_targ = X_batch, Y_batch

        # Forward pass (FP)
        for l in range(1, len(self.layers)):
            self.tracer.emit_event(PYDTNN_MDL_EVENT, self.layers[l].id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_FORWARD)
            x = self.layers[l].forward(x)
            self.tracer.emit_event(PYDTNN_MDL_EVENT, 0)

        loss, dx = loss_func(x, Y_targ, global_batch_size)
        self.total_metrics, _ = self.__compute_metrics_funcs(x, Y_targ, loss, metrics_funcs)

        if self.blocking_mpi:
            # Blocking MPI
            # Back propagation. Gradient computation (GC) and weights update (WU)
            for l in range(len(self.layers) - 1, 0, -1):
                # self.tracer.print_memory_usage(f"Layer {l:03} before backward")
                self.tracer.emit_event(PYDTNN_MDL_EVENT, self.layers[l].id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_BACKWARD)
                dx = self.layers[l].backward(dx)
                # self.tracer.print_memory_usage(f"Layer {l:03} after backward ")
                self.tracer.emit_event(PYDTNN_MDL_EVENT, 0)

            if self.enable_cudnn:
                self.stream.synchronize()

            # Weight update (WU)
            for l in range(len(self.layers) - 1, 0, -1):
                self.layers[l].reduce_weights_sync()
                self.tracer.emit_event(PYDTNN_MDL_EVENT, self.layers[l].id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_UPDATE_DW)
                self.layers[l].update_weights(optimizer)
                self.tracer.emit_event(PYDTNN_MDL_EVENT, 0)
        else:
            # Non-blocking MPI
            # Back propagation. Gradient computation (GC) and weights update (WU)
            for l in range(len(self.layers) - 1, 0, -1):
                self.tracer.emit_event(PYDTNN_MDL_EVENT, self.layers[l].id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_BACKWARD)
                dx = self.layers[l].backward(dx)
                self.tracer.emit_event(PYDTNN_MDL_EVENT, 0)

                self.tracer.emit_event(PYDTNN_MDL_EVENT,
                                       self.layers[l].id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_ALLREDUCE_DW)
                self.layers[l].reduce_weights_async()
                self.tracer.emit_event(PYDTNN_MDL_EVENT, 0)

            # Weight update (WU)
            for l in range(len(self.layers) - 1, 0, -1):
                self.tracer.emit_nevent([PYDTNN_MDL_EVENT, PYDTNN_OPS_EVENT],
                                        [self.layers[l].id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_WAIT_DW,
                                         self.layers[l].id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_ALLREDUCE_DW])
                self.layers[l].wait_allreduce_async()
                self.tracer.emit_nevent([PYDTNN_MDL_EVENT, PYDTNN_OPS_EVENT], [0, 0])

                self.tracer.emit_event(PYDTNN_MDL_EVENT, self.layers[l].id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_UPDATE_DW)
                self.layers[l].update_weights(optimizer)
                self.tracer.emit_event(PYDTNN_MDL_EVENT, 0)

        if self.enable_cudnn:
            for l in range(len(self.layers) - 1, 0, -1):
                if self.layers[l].grad_vars:
                    self.layers[l].stream_2.synchronize()

        for lr_sched in lr_schedulers:
            lr_sched.on_batch_end(self, optimizer, self.rank)

        return self.total_metrics

    def train(self, X_train, Y_train, X_val, Y_val, nepochs, local_batch_size,
              loss="categorical_cross_entropy", metrics=["categorical_accuracy"],
              optimizer=NN_optimizer.SGD(), bar_width=110):

        dataset = datasets.NN_dataset.Dataset(X_train=X_train, Y_train=Y_train,
                                              X_val=X_val, Y_val=Y_val)
        history = self.train_dataset(dataset, nepochs, local_batch_size, 0,
                                     loss=loss, metrics=metrics, optimizer=optimizer,
                                     bar_width=bar_width)
        return history

    def train_dataset(self, dataset, nepochs, local_batch_size, val_split=0.2,
                      loss="categorical_cross_entropy", metrics=["categorical_accuracy"],
                      optimizer=NN_optimizer.SGD(), lr_schedulers=[], bar_width=110):
        if self.enable_cudnn and not hasattr(self, "Y_batch"):
            self.Y_batch = NN_util.TensorGPU(gpuarray.empty((local_batch_size, *self.layers[-1].shape), self.dtype),
                                             self.tensor_fmt, self.cudnn_dtype)
        loss_func = getattr(NN_util, loss)(shape=(local_batch_size, *self.layers[-1].shape), model=self,
                                           enable_gpu=self.enable_cudnn, dtype=self.dtype)
        metrics_funcs = [getattr(NN_util, l)(shape=(local_batch_size, *self.layers[-1].shape), model=self,
                                             enable_gpu=self.enable_cudnn, dtype=self.dtype) for l in metrics]
        loss_metrics = [loss] + metrics
        self.history = {l: [] for l in (loss_metrics + [f"val_{m}" for m in loss_metrics])}
        optimizer.gpudirect = self.gpudirect

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

            for X_batch, Y_batch, batch_size in train_batch_generator:
                tic = timer()
                train_batch_loss = self.__train_batch(X_batch, Y_batch, local_batch_size, batch_size,
                                                      loss_func, metrics_funcs, optimizer, lr_schedulers)
                toc = timer()
                train_total_loss, train_batch_count, string = \
                    self.__update_running_average(train_batch_loss, train_total_loss,
                                                  train_batch_count, batch_size,
                                                  loss_metrics)
                if self.rank == 0:
                    pbar.set_postfix_str(s=string, refresh=True)
                    pbar.update(batch_size)
                    self.perf_counter.add_training_time_and_batch_size(epoch, toc - tic, batch_size)

            if self.rank == 0:
                pbar.close()
                for c in range(len(loss_metrics)):
                    self.history[loss_metrics[c]].append(train_total_loss[c])

            for X_batch, Y_batch, batch_size in val_batch_generator:
                val_batch_loss = self.__evaluate_batch(X_batch, Y_batch, local_batch_size, batch_size,
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

    def __evaluate_batch(self, X_batch, Y_batch, local_batch_size, global_batch_size, loss_func, metrics_funcs):
        self.mode = "evaluate"

        if self.enable_cudnn:
            if X_batch.shape[0] != local_batch_size:
                return self.total_metrics
            self.layers[0].y.ary.set(X_batch)
            self.Y_batch.ary.set(Y_batch)
            x, Y_targ = self.layers[0].y, self.Y_batch
        else:
            x, Y_targ = X_batch, Y_batch

        # Forward pass (FP)
        for l in range(1, len(self.layers)):
            self.tracer.emit_event(PYDTNN_MDL_EVENT, self.layers[l].id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_FORWARD)
            x = self.layers[l].forward(x)
            self.tracer.emit_event(PYDTNN_MDL_EVENT, 0)

        Y_pred = self.layers[-1].y
        loss, _ = loss_func(Y_pred, Y_targ, global_batch_size)
        self.total_metrics, _ = self.__compute_metrics_funcs(Y_pred, Y_targ, loss, metrics_funcs)
        return self.total_metrics

    def evaluate(self, X_test, Y_test, local_batch_size,
                 loss="categorical_cross_entropy", metrics=["categorical_accuracy"],
                 bar_width=110):

        dataset = datasets.NN_dataset.Dataset(X_test=X_test, Y_test=Y_test)
        self.evaluate_dataset(dataset, local_batch_size, loss=loss, metrics=metrics, bar_width=bar_width)

    def evaluate_dataset(self, dataset, local_batch_size,
                         loss="categorical_cross_entropy", metrics=["categorical_accuracy"],
                         bar_width=120):
        if self.enable_cudnn and not hasattr(self, "Y_batch"):
            self.Y_batch = NN_util.TensorGPU(gpuarray.empty((local_batch_size, *self.layers[-1].shape), self.dtype),
                                             self.tensor_fmt, self.cudnn_dtype)
        loss_func = getattr(NN_util, loss)(shape=(self.params.batch_size, *self.layers[-1].shape), model=self,
                                           enable_gpu=self.enable_cudnn, dtype=self.dtype)
        metrics_funcs = [getattr(NN_util, l)(shape=(self.params.batch_size, *self.layers[-1].shape), model=self,
                                             enable_gpu=self.enable_cudnn, dtype=self.dtype) for l in metrics]
        loss_metrics = [loss] + metrics
        test_batch_generator = dataset.get_test_generator(self.rank, self.nprocs)

        if self.rank == 0:
            test_total_loss, test_batch_count = np.zeros(len(loss_metrics)), 0
            pbar = tqdm(total=dataset.test_nsamples, ncols=bar_width,
                        ascii=" ▁▂▃▄▅▆▇█", smoothing=0.3,
                        desc="Testing", unit=" samples")

        self.perf_counter.init_testing_data()  # Only the last testing data should be kept

        for X_batch, Y_batch, batch_size in test_batch_generator:
            tic = timer()
            test_batch_loss = self.__evaluate_batch(X_batch, Y_batch, self.params.batch_size, batch_size,
                                                    loss_func, metrics_funcs)
            toc = timer()
            if self.rank == 0:
                val_total_loss, val_batch_count, string = \
                    self.__update_running_average(test_batch_loss, test_total_loss, test_batch_count, batch_size,
                                                  loss_metrics, prefix="test_")
                pbar.set_postfix_str(s=string, refresh=True)
                pbar.update(batch_size)
                self.perf_counter.add_testing_time_and_batch_size(toc - tic, batch_size)

        if self.rank == 0:
            pbar.close()
            # Sleep for half a second to allow pbar to write its output before returning
            time.sleep(.5)
