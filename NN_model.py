""" Python Distributed Training of Neural Networks - PyDTNN

PyDTNN is a light-weight library for distributed Deep Learning training and 
inference that offers an initial starting point for interaction with 
distributed training of (and inference with) deep neural networks. PyDTNN 
priorizes simplicity over efficiency, providing an amiable user interface 
which enables a flat accessing curve. To perform the training and inference 
processes, PyDTNN exploits distributed inter-process parallelism (via MPI) 
for clusters and intra-process (via multi-threading) parallelism to leverage 
the presence of multicore processors at node level.

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
__version__ = "1.0.0"


import numpy as np

import random, sys
import NN_util, NN_activation, NN_optimizer, datasets.NN_dataset
from NN_tracer import Tracer, PYDL_EVT, PYDL_OPS_EVT, PYDL_NUM_EVTS, \
                              PYDL_OPS_EVT, PYDL_OPS_NUM_EVTS
from tqdm import tqdm

supported_gpu = False
supported_mpi4py = False
try:
    import pycuda.autoinit
    import pycuda.gpuarray as gpuarray
    import pycuda.driver as drv
    import skcuda.linalg as culinalg
    import skcuda.misc as cumisc
    supported_gpu = True
except:
    pass

try:
    from mpi4py import MPI
    supported_mpi4py = True
except:
    pass

class Model:
    """ Neural network (NN) """

    def __init__(self, params, comm=None, non_blocking_mpi=False, 
                 tracing=False, enable_gpu=False, dtype=np.float32):
        self.layers = []
        self.params = params
        self.comm = comm
        self.blocking_mpi = not non_blocking_mpi
        self.tracer = Tracer(tracing)
        self.enable_gpu = enable_gpu
        self.dtype = dtype

        self.rank = 0
        self.nprocs = 1

        if self.comm != None and supported_mpi4py:
            self.rank = self.comm.Get_rank()
            self.nprocs = self.comm.Get_size()
        elif self.comm != None:
            print("You must install mpi4py to allow parallel MPI execution!")
            sys.exit(-1)

        if self.enable_gpu and supported_gpu:
            culinalg.init()
        elif self.enable_gpu:
            print("You must install pycuda+skcuda to allow parallel MPI execution!")
            sys.exit(-1)

    def show(self):
        print("+-------+----------+---------+---------------+-----------------+---------+---------+")
        print("| Layer |   Type   | #Params | Output shape  |  Weights shape  | Padding | Stride  |")
        for l in self.layers:
            print('+-------+----------+---------+---------------+-----------------+---------+---------+')
            l.show()
        print('+-------+----------+---------+---------------+-----------------+---------+---------+')

        # print("┌───────┬──────────┬─────────┬───────────────┬─────────────────┬─────────┬─────────┐")
        # print("│ Layer │   Type   │ #Params │ Output shape  │  Weights shape  │ Padding │ Stride  │")
        # for l in self.layers:
        #     print('├───────┼──────────┼─────────┼───────────────┼─────────────────┼─────────┼─────────┤')
        #     l.show()
        # print('└───────┴──────────┴─────────┴───────────────┴─────────────────┴─────────┴─────────┘')

    def add(self, layer):
        layer.id = len(self.layers)
        layer.tracer = self.tracer
        layer.dtype = self.dtype
        layer.matmul = getattr(NN_util, {False: "matmul", True: "matmul_gpu"}[self.enable_gpu])

        if len(self.layers) > 0:          
            self.layers[-1].next_layer = layer
            layer.prev_layer = self.layers[-1]
            layer.initialize()

        self.layers.append(layer)
        if layer.act:
            layer.act.shape = layer.shape
            self.add(layer.act)

    def compute_loss_funcs(self, Y_pred, Y_targ, loss_funcs, blocking=True):
        loss_req = None
        total_loss = np.zeros(len(loss_funcs), dtype=np.float32)
        partial_loss = np.array([func(Y_pred, Y_targ) for func in loss_funcs], dtype=np.float32)
        if self.comm != None and blocking:
            self.comm.Reduce(partial_loss, total_loss, op=MPI.SUM, root=0)
            total_loss /= self.nprocs
        elif self.comm != None and not blocking:
            loss_req = self.comm.Ireduce(partial_loss, total_loss, op=MPI.SUM)
        else:
            total_loss = partial_loss
        return total_loss, loss_req

    def get_metric_results(self, curr, loss):
        total, count, string = 
            self.__update_running_average(curr, np.zeros(len(loss)), 0, loss, prefix="test_")
        return string

    def __update_running_average(self, curr, total, count, loss_metrics, prefix=""):
        string = ""
        total = (curr + (total * count)) / (count+1)
        for c in range(len(total)):
            try:    loss_str = NN_util.loss_format[loss_metrics[c]]
            except: loss_str = loss_metrics[c]
            string += ("%s, " % (prefix+loss_str)) % total[c]
        string = string[:-2]
        return total, count+1, string

    def __train_batch(self, X_batch, Y_batch, loss_funcs, optimizer_func):

        if X_batch.shape[0] == 0: return [0]

        # Forward pass (FP)
        self.layers[0].a = X_batch
        for l in range(1, len(self.layers)):
            self.tracer.emit_event(PYDL_EVT, self.layers[l].id * PYDL_NUM_EVTS + 1)
            self.layers[l].forward(self.layers[l-1].a)
            self.tracer.emit_event(PYDL_EVT, 0)

        Y_pred = self.layers[-1].a
        total_loss, loss_req = self.compute_loss_funcs(Y_pred, Y_batch, loss_funcs, blocking=False)
       
        if self.blocking_mpi:
            # Blocking MPI
            # Back propagation. Gradient computation (GC) and weights update (WU)
            dx = (Y_pred - Y_batch)
            for l in range(len(self.layers)-1, 0, -1):
                self.tracer.emit_event(PYDL_EVT, self.layers[l].id * PYDL_NUM_EVTS + 2)
                dx = self.layers[l].backward(dx)
                self.tracer.emit_event(PYDL_EVT, 0)
    
            # Weight update (WU)
            for l in range(len(self.layers)-1, 0, -1):
                self.layers[l].reduce_weights_sync(self.comm)
                self.tracer.emit_event(PYDL_EVT, self.layers[l].id * PYDL_NUM_EVTS + 5)
                self.layers[l].update_weights(optimizer_func, self.params)
                self.tracer.emit_event(PYDL_EVT, 0)            
        else:
            # Non-blocking MPI
            # Back propagation. Gradient computation (GC) and weights update (WU)
            dx = (self.layers[-1].a - Y_batch)
            for l in range(len(self.layers)-1, 0, -1):
                self.tracer.emit_event(PYDL_EVT, self.layers[l].id * PYDL_NUM_EVTS + 2)
                dx = self.layers[l].backward(dx)
                self.tracer.emit_event(PYDL_EVT, 0)

                self.tracer.emit_event(PYDL_EVT, self.layers[l].id * PYDL_NUM_EVTS + 3)
                self.layers[l].reduce_weights_async(self.comm)
                self.tracer.emit_event(PYDL_EVT, 0)
    
            # Weight update (WU)
            for l in range(len(self.layers)-1, 0, -1):
                self.tracer.emit_nevent([PYDL_EVT, PYDL_OPS_EVT], 
                                        [self.layers[l].id * PYDL_NUM_EVTS + 4, 
                                         self.layers[l].id * PYDL_OPS_NUM_EVTS + 6])
                self.layers[l].wait_allreduce_async(self.comm)
                self.tracer.emit_nevent([PYDL_EVT, PYDL_OPS_EVT], [0, 0])
    
                self.tracer.emit_event(PYDL_EVT, self.layers[l].id * PYDL_NUM_EVTS + 5)
                self.layers[l].update_weights(optimizer_func, self.params)
                self.tracer.emit_event(PYDL_EVT, 0)

        if self.comm != None:
            #if self.rank == 0: print("total: ", total_loss)
            loss_req.Wait()
            total_loss /= self.nprocs
            #if self.rank == 0: print("total: ", total_loss)

        return total_loss

    def train(self, X_train, Y_train, X_val, Y_val, nepochs, local_batch_size,
                    loss_metrics=["categorical_accuracy", "categorical_cross_entropy"], 
                    optimizer="SGD", bar_width=110):

        dataset = datasets.NN_dataset.Dataset(X_train=X_train, Y_train=Y_train, 
                                              X_val=X_val, Y_val=Y_val)
        self.train_dataset(dataset, nepochs, local_batch_size, 0, False,
                           loss_metrics, optimizer, bar_width)

    def train_dataset(self, dataset, nepochs, local_batch_size, 
                      val_split=0.2, use_test_as_validation=False,
                      loss_metrics=["categorical_accuracy", "categorical_cross_entropy"], 
                      optimizer="SGD", bar_width=110):

        loss_funcs = [getattr(NN_util, l) for l in loss_metrics]
        optimizer_func = getattr(NN_optimizer, optimizer)
        
        for epoch in range(nepochs):

            train_batch_generator, val_batch_generator = \
                dataset.get_train_val_generator(local_batch_size, self.rank, self.nprocs, val_split)

            if self.rank == 0:
                train_total_loss, train_batch_count = np.zeros(len(loss_funcs)), 0
                val_total_loss, val_batch_count = np.zeros(len(loss_funcs)), 0
                fmt="%%%dd" % (len(str(nepochs)))
                epoch_string="Epoch %s/%s" % (fmt, fmt)
                pbar = tqdm(total=dataset.train_nsamples, ncols=bar_width, 
                            ascii=" ▁▂▃▄▅▆▇█", smoothing=0.3,
                            desc=epoch_string % (epoch+1, nepochs), unit=" samples")

            for X_batch, Y_batch, batch_size in train_batch_generator:
                train_batch_loss = self.__train_batch(X_batch, Y_batch, loss_funcs, optimizer_func)
                if self.rank == 0:
                    train_total_loss, train_batch_count, string = \
                        self.__update_running_average(train_batch_loss, train_total_loss, 
                                                      train_batch_count, loss_metrics)
                    pbar.set_postfix_str(s=string, refresh=True)
                    pbar.update(batch_size)

            if self.rank == 0:
                pbar.close()

            for X_batch, Y_batch, batch_size in val_batch_generator:
                val_batch_loss = self.__evaluate_batch(X_batch, Y_batch, loss_funcs)
                if self.rank == 0 and X_batch.shape[0] > 0:
                    val_total_loss, val_batch_count, string = \
                        self.__update_running_average(val_batch_loss, val_total_loss, 
                                                      val_batch_count, loss_metrics, 
                                                      prefix="val_")
                    print("\033[A\033[%dC\b, %s]" % (bar_width, string))

        self.tracer.define_event_type()

    def __evaluate_batch(self, X_batch, Y_batch, loss_funcs):

        # Forward pass (FP)
        if X_batch.shape[0] == 0: return [0]

        self.layers[0].a = X_batch
        for l in range(1, len(self.layers)):
            self.tracer.emit_event(PYDL_EVT, self.layers[l].id * 7 + 2)
            self.layers[l].forward(self.layers[l-1].a)
            self.tracer.emit_event(PYDL_EVT, 0)

        Y_pred = self.layers[-1].a
        loss_res, loss_reqs = self.compute_loss_funcs(Y_pred, Y_batch, loss_funcs, blocking=True)
        return loss_res

    def evaluate(self, X_test, Y_test, 
                 loss_metrics=["categorical_accuracy", "categorical_cross_entropy"], 
                 bar_width=110):

        dataset = datasets.NN_dataset.Dataset(X_test=X_test, Y_test=Y_test)
        self.evaluate_dataset(dataset, loss_metrics, bar_width)

    def evaluate_dataset(self, dataset, 
                         loss_metrics=["categorical_accuracy", "categorical_cross_entropy"], 
                         bar_width=120):

        loss_funcs = [getattr(NN_util, l) for l in loss_metrics]
        test_batch_generator = dataset.get_test_generator(self.rank, self.nprocs)

        if self.rank == 0:
            test_total_loss, test_batch_count = np.zeros(len(loss_funcs)), 0
            pbar = tqdm(total=dataset.test_nsamples, ncols=bar_width, 
                        ascii=" ▁▂▃▄▅▆▇█", smoothing=0.3,
                        desc="Testing", unit=" samples")

        for X_batch, Y_batch, batch_size in test_batch_generator:
            test_batch_loss = self.__evaluate_batch(X_batch, Y_batch, loss_funcs)
            if self.rank == 0 and X_batch.shape[0] > 0:
                val_total_loss, val_batch_count, string = \
                    self.__update_running_average(test_batch_loss, test_total_loss, 
                                                  test_batch_count, loss_metrics, 
                                                  prefix="test_")
                pbar.set_postfix_str(s=string, refresh=True)
                pbar.update(batch_size)

        if self.rank == 0:
            pbar.close()

