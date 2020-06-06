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
__version__ = "1.0.1"


import numpy as np
import NN_util, NN_activation, NN_initializer

from math import floor
from NN_util import printf
from NN_im2col_cython import im2col_cython, col2im_cython
from NN_argmax_cython import argmax_cython
from NN_add_cython import add_cython
from NN_tracer import PYDL_EVT, PYDL_OPS_EVT, PYDL_NUM_EVTS, PYDL_OPS_EVT, PYDL_OPS_NUM_EVTS

try:
    from mpi4py import MPI
except:
    pass

class Layer():

    def __init__(self, shape=()):
        self.id, self.params = 0, 0
        self.shape = shape
        self.prev_layer, self.next_layer = None, None
        self.weights, self.bias = np.array([]), np.array([])
        self.act = None
        self.grad_vars = {}

    def initialize(self):
        self.shape = self.prev_layer.shape

    def show(self, attrs=""):
        if not attrs: attrs= "|{:^17s}|{:^21s}|".format("","")
        print(f"|{self.id:^7d}|{type(self).__name__:^20s}|{self.params:^9d}|{str(self.shape):^15}" + attrs)

    def update_weights(self, optimizer):
        optimizer.update(self)

    def reduce_weights_async(self, comm):
        if comm and self.weights.size > 0:
            self.dwb = np.concatenate((self.dw.flatten(), self.db.flatten()))
            self.req_AR = comm.Iallreduce(MPI.IN_PLACE, self.dwb, op=MPI.SUM)
     
    def wait_allreduce_async(self, comm):
        if comm and self.weights.size > 0:
            self.req_AR.Wait()
            self.dw = self.dwb[:self.weights.size].reshape(self.weights.shape)
            self.db = self.dwb[self.weights.size:].reshape(self.bias.shape)
     
    def reduce_weights_sync(self, comm):
        if comm and self.weights.size > 0:
            dwb = np.concatenate((self.dw.flatten(), self.db.flatten()))
            self.tracer.emit_nevent([PYDL_EVT, PYDL_OPS_EVT], 
                                    [self.id * PYDL_NUM_EVTS + 3, 
                                     self.id * PYDL_OPS_NUM_EVTS + 6])
            comm.Allreduce(MPI.IN_PLACE, dwb, op=MPI.SUM)
            self.tracer.emit_nevent([PYDL_EVT, PYDL_OPS_EVT], [0, 0])
            self.dw = dwb[:self.weights.size].reshape(self.weights.shape)
            self.db = dwb[self.weights.size:].reshape(self.bias.shape)


class Input(Layer):

    def __init__(self, shape=(1,)):
        super().__init__(shape)


class FC(Layer):

    def __init__(self, shape=(1,), activation="", 
                 weights_initializer="glorot_uniform",
                 bias_initializer="zeros"):
        super().__init__(shape)
        self.act = getattr(NN_activation, activation, None)
        self.weights_initializer = getattr(NN_initializer, weights_initializer)
        self.bias_initializer = getattr(NN_initializer, bias_initializer)
        self.grad_vars = {"weights": "dw", "bias": "db"}
        
    def initialize(self):
        self.weights = self.weights_initializer((np.prod(self.prev_layer.shape), 
                                                 np.prod(self.shape[0])), self.dtype)
        self.bias = self.bias_initializer((np.prod(self.shape),), self.dtype)
        self.params = np.prod(self.weights.shape) + np.prod(self.bias.shape)
        
    def show(self):
        super().show("|{:^17s}|{:^21s}|".format(str(self.weights.shape),""))

    def forward(self, prev_a, comm=None):
        self.tracer.emit_event(PYDL_OPS_EVT, self.id * PYDL_OPS_NUM_EVTS + 1)
        res = self.matmul(prev_a, self.weights)
        self.tracer.emit_event(PYDL_OPS_EVT, 0)
        self.a = res + self.bias
        
    def backward(self, prev_dx):
        self.tracer.emit_event(PYDL_OPS_EVT, self.id * PYDL_OPS_NUM_EVTS + 3)
        dx = self.matmul(prev_dx, self.weights.T)
        self.tracer.emit_event(PYDL_OPS_EVT, 0)

        self.tracer.emit_event(PYDL_OPS_EVT, self.id * PYDL_OPS_NUM_EVTS + 5)
        self.dw = self.matmul(self.prev_layer.a.T, prev_dx)
        self.tracer.emit_event(PYDL_OPS_EVT, 0)
        self.db = prev_dx.sum(axis=0)
        return dx
        

class Conv2D(Layer):

    def __init__(self, nfilters=1, filter_shape=(3, 3), padding=0, stride=1, 
                 activation="", weights_initializer="glorot_uniform",
                 bias_initializer="zeros"):
        super().__init__()
        self.co = nfilters
        self.filter_shape = filter_shape
        self.padding = padding
        self.stride = stride
        self.act = getattr(NN_activation, activation, None)
        self.weights_initializer = getattr(NN_initializer, weights_initializer)
        self.bias_initializer = getattr(NN_initializer, bias_initializer)
        self.grad_vars = {"weights": "dw", "bias": "db"}

    def initialize(self):
        self.ci, self.hi, self.wi = self.prev_layer.shape
        self.kh, self.kw = self.filter_shape

        self.weights = self.weights_initializer(((self.co,)+(self.ci,)+self.filter_shape), self.dtype)
        self.bias = self.bias_initializer((self.co,), self.dtype)

        self.ho = floor((self.hi + 2 * self.padding - self.kh) / self.stride) + 1
        self.wo = floor((self.wi + 2 * self.padding - self.kw) / self.stride) + 1
        self.shape = (self.co, self.ho, self.wo)
        self.params = np.prod(self.weights.shape) + np.prod(self.bias.shape)

    def show(self):
        super().show("|{:^17s}|{:^21s}|".format(str(self.weights.shape), \
            "padding=%d, stride=%d" % (self.padding, self.stride)))

    def forward(self, prev_a, comm=None):
        self.tracer.emit_event(PYDL_OPS_EVT, self.id * PYDL_OPS_NUM_EVTS + 2)
        self.prev_a_cols = im2col_cython(prev_a, self.kh, self.kw, self.padding, self.stride)
        self.tracer.emit_event(PYDL_OPS_EVT, 0)

        w_cols = self.weights.reshape(self.co, -1)
        self.tracer.emit_event(PYDL_OPS_EVT, self.id * PYDL_OPS_NUM_EVTS + 1)
        res = self.matmul(w_cols, self.prev_a_cols)
        self.tracer.emit_event(PYDL_OPS_EVT, 0)

        # a = res + self.bias.reshape(-1, 1)
        a = add_cython(res, self.bias)
        self.a = a.reshape(self.co, -1, self.ho, self.wo).transpose(1, 0, 2, 3)

    def backward(self, prev_dx):
        dx_cols = prev_dx.transpose(1, 0, 2, 3).reshape(self.co, -1)
        w_cols = self.weights.reshape(self.co, -1).T
        
        self.tracer.emit_event(PYDL_OPS_EVT, self.id * PYDL_OPS_NUM_EVTS + 3)
        res = self.matmul(w_cols, dx_cols)
        self.tracer.emit_event(PYDL_OPS_EVT, 0)

        self.tracer.emit_event(PYDL_OPS_EVT, self.id * PYDL_OPS_NUM_EVTS + 4)
        dx = col2im_cython(res, prev_dx.shape[0], self.ci, self.hi, self.wi, 
                                   self.kh, self.kw, self.padding, self.stride)
        self.tracer.emit_event(PYDL_OPS_EVT, 0)

        self.tracer.emit_event(PYDL_OPS_EVT, self.id * PYDL_OPS_NUM_EVTS + 5)
        res = self.matmul(dx_cols, self.prev_a_cols.T)
        self.tracer.emit_event(PYDL_OPS_EVT, 0)

        self.dw = res.reshape(self.weights.shape)
        self.db = prev_dx.sum(axis=(0,2,3))

        return dx


class MaxPool2D(Layer):

    def __init__(self, pool_shape=(2,2), padding=0, stride=1):
        super().__init__()
        self.pool_shape = pool_shape
        self.padding = padding
        self.stride = stride     

    def initialize(self):
        self.ci, self.hi, self.wi = self.prev_layer.shape
        self.kh, self.kw = self.pool_shape
        self.ho = floor((self.hi - self.kh) / self.stride) + 1
        self.wo = floor((self.wi - self.kw) / self.stride) + 1
        self.co = self.ci
        self.shape = (self.co, self.ho, self.wo)
        self.n = np.prod(self.shape)

    def show(self):
        super().show("|{:^17s}|{:^21s}|".format(str(self.pool_shape), \
            "padding=%d, stride=%d" % (self.padding, self.stride)))

    def forward(self, prev_a, comm=None):
        prev_a_ = prev_a.reshape(prev_a.shape[0] * self.ci, 1, self.hi, self.wi)
        self.tracer.emit_event(PYDL_OPS_EVT, self.id * PYDL_OPS_NUM_EVTS + 2)
        a_cols = im2col_cython(prev_a_, self.kh, self.kw, self.padding, self.stride)
        self.tracer.emit_event(PYDL_OPS_EVT, 0)

        # self.maxids = tuple([np.argmax(a_cols, axis=0), np.arange(a_cols.shape[1])])
        self.a, self.maxids = argmax_cython(a_cols, axis=0)        
        self.a = self.a.reshape(prev_a.shape[0], self.co, self.ho, self.wo)

    def backward(self, prev_dx):
        dx_cols = np.zeros((self.kh * self.kw, np.prod(prev_dx.shape)), dtype=self.dtype)
        dx_cols[self.maxids] = prev_dx.flatten()
        self.tracer.emit_event(PYDL_OPS_EVT, self.id * PYDL_OPS_NUM_EVTS + 4)
        dx = col2im_cython(dx_cols, prev_dx.shape[0] * self.ci, 1, self.hi, self.wi, 
                           self.kh, self.kw, self.padding, self.stride)
        self.tracer.emit_event(PYDL_OPS_EVT, 0)
        dx = dx.reshape(prev_dx.shape[0], self.ci, self.hi, self.wi)
        return dx


class Dropout(Layer):

    def __init__(self, rate=0.5):
        super().__init__()
        self.rate = min(1., max(0., rate))

    def initialize(self):
        self.shape = self.prev_layer.shape

    def show(self):
        super().show("|{:^17s}|{:^21s}|".format("", "rate=%.1f" % (self.rate)))

    def forward(self, prev_a, comm=None):
        if self.model.mode == "train":
            self.mask = np.random.binomial(1, (1-self.rate), size=self.shape).astype(self.dtype) / (1-self.rate)
            self.a = prev_a * self.mask

        elif self.model.mode == "evaluate":
            self.a = prev_a

    def backward(self, prev_dx):
        return prev_dx * self.mask
 

class Flatten(Layer):

    def __init__(self):
        super().__init__()

    def initialize(self):
        self.shape = (np.prod(self.prev_layer.shape),)
        self.n = np.prod(self.shape)

    def forward(self, prev_a, comm=None):
        self.a = prev_a.reshape(prev_a.shape[0], -1)

    def backward(self, prev_dx):
        return prev_dx.reshape((prev_dx.shape[0],) + self.prev_layer.shape)


class BatchNormalization(Layer):

    def __init__(self, beta=0.0, gamma=1.0, 
                 momentum=0.9, epsilon=1e-5,
                 moving_mean_initializer="zeros",
                 moving_variance_initializer="ones",
                 sync_stats=False):
        super().__init__()
        self.gamma_init_val = gamma
        self.beta_init_val = beta
        self.momentum = momentum
        self.epsilon = epsilon
        self.moving_mean_initializer = getattr(NN_initializer, moving_mean_initializer)
        self.moving_variance_initializer = getattr(NN_initializer, moving_variance_initializer)
        self.grad_vars = {"beta": "dbeta", "gamma": "dgamma"}
        self.sync_stats = sync_stats

    def initialize(self):
        self.shape = shape_ = self.prev_layer.shape
        self.spatial = len(self.shape) > 2
        if self.spatial:
            self.co = self.ci = self.shape[0]
            self.hi, self.wi = self.shape[1], self.shape[2]
            shape_ = (self.ci)
        self.beta = np.full(shape_, self.beta_init_val, self.dtype)
        self.gamma = np.full(shape_, self.gamma_init_val, self.dtype)
        self.running_mean = self.moving_mean_initializer(shape_, self.dtype)
        self.running_var = self.moving_variance_initializer(shape_, self.dtype)

    def forward(self, prev_a, comm=None):

        def mean(data, N, comm):
            if self.sync_stats and comm != None:
                mean = np.sum(data, axis=0) / N
                comm.Allreduce(MPI.IN_PLACE, mean, op=MPI.SUM)
            else:
                mean = np.mean(data, axis=0)
            return mean

        if self.spatial:
            prev_a = prev_a.transpose(0, 2, 3, 1).reshape(-1, self.ci)
        
        if self.model.mode == "train":
            N = np.array([prev_a.shape[0]], dtype=self.dtype)
            if self.sync_stats and comm != None:
                comm.Allreduce(MPI.IN_PLACE, N, op=MPI.SUM)

            mu = mean(prev_a, N, comm)
            xc = (prev_a - mu)
            var = mean(xc**2, N, comm)

            self.std = np.sqrt(var + self.epsilon)
            self.xn = xc / self.std
            self.a = self.gamma * self.xn + self.beta
  
            self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * var

        elif self.model.mode == "evaluate":
            std = np.sqrt(self.running_var + self.epsilon)
            xn = (prev_a - self.running_mean) / std
            self.a = self.gamma * xn + self.beta

        if self.spatial:
            self.a = self.a.reshape(-1, self.hi, self.wi, self.ci).transpose(0, 3, 1, 2)

    def backward(self, prev_dx):
        if self.spatial:          
            prev_dx = prev_dx.transpose(0, 2, 3, 1).reshape(-1, self.ci)

        N = prev_dx.shape[0]
        self.dgamma = np.sum(prev_dx * self.xn, axis=0)
        self.dbeta = np.sum(prev_dx, axis=0)
        dx = (self.gamma / (self.std * N)) * (N * prev_dx - self.xn * self.dgamma - self.dbeta)
        dx = dx.astype(self.dtype)

        if self.spatial:
            dx = dx.reshape(-1, self.hi, self.wi, self.ci).transpose(0, 3, 1, 2)
        return dx
