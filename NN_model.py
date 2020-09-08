""" Python Distributed Training of Neural Networks - PyDTNN

PyDTNN is a light-weight library for distributed Deep Learning training and 
inference that offers an initial starting point for interaction with 
distributed training of (and inference with) deep neural networks. PyDTNN 
priorizes simplicity over efficiency, providing an amiable user interface 
which enables a flat accessing curve. To perform the training and inference 
processes, PyDTNN exploits distributed inter-process parallelism (via MPI) 
for clusters and intra-process (via multi-threading) parallelism to leverage 
the presence of multicore processors and GPUs at node level. For that, PyDTNN 
uses MPI4Py for message-passing, BLAS calls via NumPy for multicore processors
and PyCUDA+cuDNN+cuBLAS for NVIDIA GPUs.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.

"""

__author__ = "Manuel F. Dolz, Enrique S. Quintana, \
              Mar Catalan, Adrian Castello"
__contact__ = "dolzm@uji.es"
__copyright__ = "Copyright 2020, Universitat Jaume I"
__credits__ = ["Manuel F. Dolz, Enrique S. Quintana", \
               "Mar Catalan", "Adrian Castello"]
__date__ = "2020/03/22"

__email__ =  "dolzm@uji.es"
__license__ = "GPLv3"
__maintainer__ = "Manuel F. Dolz"
__status__ = "Production"
__version__ = "1.1.0"


import numpy as np

import random, sys, os
import NN_activation
import NN_util 
import NN_optimizer
import datasets.NN_dataset
from NN_tracer import Tracer, PYDL_EVT, PYDL_OPS_EVT, PYDL_NUM_EVTS, \
                              PYDL_OPS_EVT, PYDL_OPS_NUM_EVTS
from NN_sim import *
from tqdm import tqdm

supported_gpu = False
supported_cudnn = False
supported_mpi4py = False
enable_cudnn = False
try:
    import pycuda.gpuarray as gpuarray
    import pycuda.driver as drv
    import libcudnn.libcudnn as cudnn
    import libnccl.libnccl as nccl
    from skcuda import cublas
    supported_cudnn = True
except Exception as e:
    print(e)

try:
    from mpi4py import MPI
    supported_mpi4py = True
except Exception as e:
    print(e)

class Model:

    def __init__(self, params, comm=None, non_blocking_mpi=False, 
                 tracing=False, enable_gpu=False, enable_gpudirect=False,
                 enable_nccl=False, dtype=np.float32):
        self.layers = []
        self.params = params
        self.comm = comm
        self.blocking_mpi = not non_blocking_mpi
        self.tracer = Tracer(tracing)
        self.id = 0
        self.enable_cudnn = enable_gpu
        global enable_cudnn
        enable_cudnn = self.enable_cudnn
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

        if self.enable_cudnn and supported_cudnn:
            import pycuda.autoinit
            # # Use this code if autoinit is not available
            # device_id = self.rank % drv.Device.count()
            # drv.init()
            # context = drv.Device(device_id).make_context()
            # import atexit
            # atexit.register(context.pop)

            if self.enable_nccl and self.comm:
                types = {np.float64: nccl.DataType.Float64,
                         np.float32: nccl.DataType.Float32,
                         np.int8:    nccl.DataType.Int8,
                         np.int32:   nccl.DataType.Int32}   
        
                try:    self.nccl_type = types[self.type]
                except: self.nccl_type = nccl.DataType.Float32
    
                hostname = MPI.Get_processor_name()
    
                hosts_data = comm.allgather([self.rank, hostname])
                # Build a dictionary hostname : [ranks_in_host]
                #   { "host1" : [0, 1], "host2" : [2, 3] }
                hosts = {}
                for r, h in hosts_data:
                   if not h in hosts: hosts[h] = [r]
                   else: hosts[h].append(r)
                
                # Check that no more processes than GPUs per node are used
                for host, ranks_in_host in hosts.items():
                   assert len(ranks_in_host) <= self.params.gpus_per_node
                
                self.intra_ranks = hosts[hostname]
                # Only a master process per node is selected as inter rank
                self.inter_ranks = [r[0] for h, r in hosts.items()]
                    
                intra_group_ = comm.Get_group()
                intra_group = MPI.Group.Incl(intra_group_, self.intra_ranks)
                intra_comm = comm.Create(intra_group)
                
                if len(self.inter_ranks) > 1:
                   inter_group_ = comm.Get_group()
                   inter_group = MPI.Group.Incl(inter_group_, self.inter_ranks)
                   self.inter_comm = comm.Create(inter_group)
                
                # Get an id once per master process and distribute it to all intra ranks
                id = intra_comm.bcast(nccl.ncclGetUniqueId() if self.rank in self.inter_ranks else None)
                self.nccl_comm = nccl.ncclCommInitRank(len(self.intra_ranks), id, intra_comm.Get_rank())            

            elif self.enable_nccl:
                self.enable_nccl = False
                print("You must install libnccl to allow NVIDIA NCCL!")
                sys.exit(-1)   

            self.cudnn_handle = cudnn.cudnnCreate()
            self.cublas_handle = cublas.cublasCreate()
            self.stream = drv.Stream()
            cublas.cublasSetStream(self.cublas_handle, self.stream.handle)
            cudnn.cudnnSetStream(self.cudnn_handle, self.stream.handle)

            types = {np.float64: "CUDNN_DATA_DOUBLE",
                     np.float32: "CUDNN_DATA_FLOAT",
                     np.int8:    "CUDNN_DATA_INT8",
                     np.int32:   "CUDNN_DATA_INT32"}

            try:    cudnn_type = types[self.type]
            except: cudnn_type = "CUDNN_DATA_FLOAT"

            self.cudnn_dtype = cudnn.cudnnDataType[cudnn_type]
            self.tensor_fmt = cudnn.cudnnTensorFormat['CUDNN_TENSOR_NCHW']

        elif self.enable_cudnn:
            self.enable_cudnn = False
            print("You must install pycuda+skcuda+cudnn to allow NVIDIA cuDNN!")
            print("or you must install pycuda+skcuda to allow GPU GEMMs executions!")
            sys.exit(-1)

    def show(self):
        bfp = {"float32": 4, "float64": 8}[self.dtype]
        print("+-------+--------------------------+---------+---------------+-------------------+------------------------+")
        print("| Layer |           Type           | #Params | Output shape  |   Weights shape   |       Parameters       |")
        for l in self.layers:
            print('+-------+--------------------------+---------+---------------+-------------------+------------------------+')
            l.show()
        print('+-------+--------------------------+---------+---------------+-------------------+------------------------+')
        print(f"|{'':^7s} {'Total parameters':^26s} {self.nparams:^9d} {NN_util.convert_size(self.nparams*bfp):^15s} {'':19s} {'':24s}|")
        print('+-------+--------------------------+---------+---------------+-------------------+------------------------+')

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
    
        if layer.act: self.add(layer.act())

    def load_weights_and_bias(self, filename):
        d = np.load(filename)
        for l, layer in enumerate(self.layers):
            base = ("%s_%s" % (str(l), type(layer).__name__))
            for p in layer.grad_vars:
                key = ("%s_%s" % (base, p))
                if key in d.files: 
                    if self.enable_cudnn: getattr(layer, p).ary.set(d[key])
                    else:                 setattr(layer, p, d[key])
                else: print("Could not find %s for layer %s in %s file!" % (p, base, filename))
                
    def store_weights_and_bias(self, filename):
        if self.params.shared_storage and self.rank == 0:
            d = {}            
            for l, layer in enumerate(self.layers):
                base = ("%s_%s" % (str(l), type(layer).__name__))
                for p in layer.grad_vars:
                    key = ("%s_%s" % (base, p))
                    d[key] = getattr(layer, p)
                    if self.enable_cudnn: d[key] = d[key].ary.get()
            np.savez_compressed(filename, **d)

    def calculate_time(self):
        time = np.zeros((4,), dtype=np.float32) # Total time, Comp time, Memo time, Net time

        # Forward pass (FP)
        for l in range(1, len(self.layers)):
            time += self.layers[l].fwd_time

        if self.blocking_mpi:
            # Blocking MPI
            # Back propagation. Gradient computation (GC) and weights update (WU)
            for l in range(len(self.layers)-1, 0, -1):
                time += self.layers[l].bwd_time
    
            # Weight update (WU)
            for l in range(len(self.layers)-1, 0, -1):
                if self.comm and self.layers[l].weights.size > 0:
                    time += allreduce_time(self.layers[l].weights.size + self.layers[l].biases.size, 
                        self.params.cpu_speed, self.params.network_bw, self.params.network_lat, 
                        self.params.network_alg, self.nprocs, self.dtype)
        else:
            time_iar = np.zeros((len(self.layers)-1, 4,), dtype=np.float32)
            # Non-blocking MPI
            # Back propagation. Gradient computation (GC) and weights update (WU)
            for l in range(len(self.layers)-1, 0, -1):
                time += self.layers[l].bwd_time
                if self.comm and self.layers[l].weights.size > 0:
                    time_iar[l] = (time_iar[l-1] if time_iar[l-1][0] > time[0] else time) if l > 0 else time + \
                        allreduce_time(self.layers[l].weights.size + self.layers[l].biases.size, 
                            self.params.cpu_speed, self.params.network_bw, self.params.network_lat, 
                            self.params.network_alg, self.nprocs, self.dtype)

            time = time_iar if time_iar[l][0] > time[0] else time

        return time

    def __compute_metrics_funcs(self, Y_pred, Y_targ, loss, metrics_funcs, blocking=True):
        loss_req = None
        if self.enable_cudnn: 
            losses = np.array([loss] + [func(Y_pred.ary, Y_targ.ary) \
                               for func in metrics_funcs], dtype=np.float32) / self.nprocs
        else:
            losses = np.array([loss] + [func(Y_pred, Y_targ) for func in metrics_funcs], dtype=np.float32) / self.nprocs
        if self.comm != None and blocking:
            self.comm.Allreduce(MPI.IN_PLACE, losses, op=MPI.SUM)
        elif self.comm != None and not blocking:
            loss_req = self.comm.Iallreduce(MPI.IN_PLACE, losses, op=MPI.SUM)
        return losses, loss_req

    def __update_running_average(self, curr, total, count, batch_size, loss_metrics, prefix=""):
        string = ""
        total = ((curr * batch_size) + (total * count)) / (count + batch_size)
        for c in range(len(loss_metrics)):
            try:    loss_str = NN_util.metric_format[loss_metrics[c]]
            except: loss_str = loss_metrics[c]
            string += ("%s, " % (prefix+loss_str)) % total[c]
        string = string[:-2]
        return total, count+batch_size, string

    def __train_batch(self, X_batch, Y_batch, local_batch_size, global_batch_size, 
                      loss_func, metrics_funcs, optimizer, lr_schedulers):

        self.mode = "train"
        for lr_sched in lr_schedulers:
            lr_sched.on_batch_begin(self, optimizer, self.rank)

        if self.enable_cudnn:
            if X_batch.shape[0] != local_batch_size: return self.total_metrics
            self.layers[0].y.ary.set(X_batch)
            self.Y_batch.ary.set(Y_batch)
            x, Y_targ = self.layers[0].y, self.Y_batch
        else:
            x, Y_targ = X_batch, Y_batch

        # Forward pass (FP)
        for l in range(1, len(self.layers)):
            self.tracer.emit_event(PYDL_EVT, self.layers[l].id * PYDL_NUM_EVTS + 1)
            x = self.layers[l].forward(x)
            self.tracer.emit_event(PYDL_EVT, 0)

        loss, dx = loss_func(x, Y_targ, global_batch_size)
        self.total_metrics, _ = self.__compute_metrics_funcs(x, Y_targ, loss, metrics_funcs)

        if self.blocking_mpi:
            # Blocking MPI
            # Back propagation. Gradient computation (GC) and weights update (WU)
            for l in range(len(self.layers)-1, 0, -1):
                self.tracer.emit_event(PYDL_EVT, self.layers[l].id * PYDL_NUM_EVTS + 2)
                dx = self.layers[l].backward(dx)
                self.tracer.emit_event(PYDL_EVT, 0)
    
            # Weight update (WU)
            for l in range(len(self.layers)-1, 0, -1):
                self.layers[l].reduce_weights_sync()
                self.tracer.emit_event(PYDL_EVT, self.layers[l].id * PYDL_NUM_EVTS + 5)
                self.layers[l].update_weights(optimizer)
                self.tracer.emit_event(PYDL_EVT, 0)            
        else:
            # Non-blocking MPI
            # Back propagation. Gradient computation (GC) and weights update (WU)
            for l in range(len(self.layers)-1, 0, -1):
                self.tracer.emit_event(PYDL_EVT, self.layers[l].id * PYDL_NUM_EVTS + 2)
                dx = self.layers[l].backward(dx)
                self.tracer.emit_event(PYDL_EVT, 0)

                self.tracer.emit_event(PYDL_EVT, self.layers[l].id * PYDL_NUM_EVTS + 3)
                self.layers[l].reduce_weights_async()
                self.tracer.emit_event(PYDL_EVT, 0)
    
            # Weight update (WU)
            for l in range(len(self.layers)-1, 0, -1):
                self.tracer.emit_nevent([PYDL_EVT, PYDL_OPS_EVT], 
                                        [self.layers[l].id * PYDL_NUM_EVTS + 4, 
                                         self.layers[l].id * PYDL_OPS_NUM_EVTS + 6])
                self.layers[l].wait_allreduce_async()
                self.tracer.emit_nevent([PYDL_EVT, PYDL_OPS_EVT], [0, 0])
    
                self.tracer.emit_event(PYDL_EVT, self.layers[l].id * PYDL_NUM_EVTS + 5)
                self.layers[l].update_weights(optimizer)
                self.tracer.emit_event(PYDL_EVT, 0)

        for lr_sched in lr_schedulers:
            lr_sched.on_batch_end(self, optimizer, self.rank)

        return self.total_metrics

    def train(self, X_train, Y_train, X_val, Y_val, nepochs, local_batch_size,
                    loss="categorical_cross_entropy", metrics=["categorical_accuracy"], 
                    optimizer=NN_optimizer.SGD(), bar_width=110):

        dataset = datasets.NN_dataset.Dataset(X_train=X_train, Y_train=Y_train, 
                                              X_val=X_val, Y_val=Y_val)
        history = self.train_dataset(dataset, nepochs, local_batch_size, 0, False,
                           loss_metrics, optimizer, bar_width)
        return history

    def train_dataset(self, dataset, nepochs, local_batch_size, 
                      val_split=0.2, loss="categorical_cross_entropy", metrics=["categorical_accuracy"], 
                      optimizer=NN_optimizer.SGD(), lr_schedulers=[], bar_width=110):
        if self.enable_cudnn and not hasattr(self, "Y_batch"):
            self.Y_batch = NN_util.TensorGPU(gpuarray.empty((local_batch_size, *self.layers[-1].shape), self.dtype),\
                                             self.tensor_fmt, self.cudnn_dtype)
        loss_func = getattr(NN_util, loss)(shape=(local_batch_size, *self.layers[-1].shape), model=self,
                                           enable_gpu=self.enable_cudnn, dtype=self.dtype)
        metrics_funcs = [getattr(NN_util, l)(shape=(local_batch_size, *self.layers[-1].shape), model=self,
                                             enable_gpu=self.enable_cudnn, dtype=self.dtype) for l in metrics]
        loss_metrics = [loss] + metrics
        self.history = {l: [] for l in (loss_metrics + ["val_%s" % m for m in loss_metrics])}
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
                fmt="%%%dd" % (len(str(nepochs)))
                epoch_string="Epoch %s/%s" % (fmt, fmt)
                pbar = tqdm(total=dataset.train_nsamples, ncols=bar_width, 
                            ascii=" ▁▂▃▄▅▆▇█", smoothing=0.3,
                            desc=epoch_string % (epoch+1, nepochs), unit=" samples")

            for lr_sched in lr_schedulers:
                lr_sched.on_epoch_begin(self, self.rank)

            for X_batch, Y_batch, batch_size in train_batch_generator:
                train_batch_loss = self.__train_batch(X_batch, Y_batch, local_batch_size, batch_size,
                                                  loss_func, metrics_funcs, optimizer, lr_schedulers)
                train_total_loss, train_batch_count, string = \
                    self.__update_running_average(train_batch_loss, train_total_loss, 
                                                  train_batch_count, batch_size, 
                                                  loss_metrics)
                if self.rank == 0:
                    pbar.set_postfix_str(s=string, refresh=True)
                    pbar.update(batch_size)

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

            if terminate: break

        self.tracer.define_event_type(self)
        return self.history

    def __evaluate_batch(self, X_batch, Y_batch, local_batch_size, global_batch_size,
                         loss_func, metrics_funcs):

        self.mode = "evaluate"
        if self.enable_cudnn:
            if X_batch.shape[0] != local_batch_size: return self.total_metrics
            self.layers[0].y.ary.set(X_batch)
            self.Y_batch.ary.set(Y_batch)
            x, Y_targ = self.layers[0].y, self.Y_batch
        else:
            x, Y_targ = X_batch, Y_batch

        # Forward pass (FP)
        for l in range(1, len(self.layers)):
            self.tracer.emit_event(PYDL_EVT, self.layers[l].id * 7 + 2)
            x = self.layers[l].forward(x)
            self.tracer.emit_event(PYDL_EVT, 0)

        Y_pred = self.layers[-1].y
        loss, _ = loss_func(Y_pred, Y_targ, global_batch_size)
        self.total_metrics, _ = self.__compute_metrics_funcs(Y_pred, Y_targ, loss, metrics_funcs)
        return self.total_metrics

    def evaluate(self, X_test, Y_test, local_batch_size,
                 loss="categorical_cross_entropy", metrics=["categorical_accuracy"],
                 bar_width=110):

        dataset = datasets.NN_dataset.Dataset(X_test=X_test, Y_test=Y_test)
        self.evaluate_dataset(dataset, local_batch_size, loss_metrics, bar_width)

    def evaluate_dataset(self, dataset, local_batch_size,
                         loss="categorical_cross_entropy", metrics=["categorical_accuracy"], 
                         bar_width=120):
        if self.enable_cudnn and not hasattr(self, "Y_batch"):
            self.Y_batch = NN_util.TensorGPU(gpuarray.empty((local_batch_size, *self.layers[-1].shape), self.dtype),\
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

        for X_batch, Y_batch, batch_size in test_batch_generator:
            test_batch_loss = self.__evaluate_batch(X_batch, Y_batch, 
                                                    self.params.batch_size, batch_size,
                                                    loss_func, metrics_funcs)
            if self.rank == 0:
                val_total_loss, val_batch_count, string = \
                    self.__update_running_average(test_batch_loss, test_total_loss, 
                                                  test_batch_count, batch_size,
                                                  loss_metrics, prefix="test_")
                pbar.set_postfix_str(s=string, refresh=True)
                pbar.update(batch_size)

        if self.rank == 0:
            pbar.close()

