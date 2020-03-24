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
import NN_util

from math import floor
from NN_util import printf
from NN_im2col_cython import im2col_cython, col2im_cython
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

    def initialize(self):
        self.shape = self.prev_layer.shape

    def show(self, attrs=""):
        if not attrs: attrs= "│{:^17s}│{:^9s}│{:^9s}│".format("","","")
        print(f"│{self.id:^7d}│{type(self).__name__:^10s}│{self.params:^9d}│{str(self.shape):^15}" + attrs)

    def update_weights(self, optimizer, params):
        if self.weights.size > 0:
            optimizer(self, params)

    def reduce_weights_async(self, comm):
        if comm and self.weights.size > 0:
            self.dwb = np.append(self.dw.reshape(-1), self.db.reshape(-1))
            self.red_dwb = np.zeros_like(self.dwb).astype(self.dtype)
            self.req_AR = comm.Iallreduce(self.dwb, self.red_dwb, op = MPI.SUM )
     
    def wait_allreduce_async(self, comm):
        if comm and self.weights.size > 0:
            self.req_AR.Wait()
            self.dw = self.red_dwb[:self.weights.size].reshape(self.weights.shape)
            self.db = self.red_dwb[-self.bias.size:].reshape(self.bias.shape)
     
    def reduce_weights_sync(self, comm):
        if comm and self.weights.size > 0:
            self.dwb = np.append(self.dw.reshape(-1), self.db.reshape(-1))
            self.red_dwb = np.zeros_like(self.dwb).astype(self.dtype)
            self.tracer.emit_nevent([PYDL_EVT, PYDL_OPS_EVT], [self.id * PYDL_NUM_EVTS + 3, self.id * PYDL_OPS_NUM_EVTS + 6])
            comm.Allreduce( self.dwb, self.red_dwb, op = MPI.SUM )
            self.tracer.emit_nevent([PYDL_EVT, PYDL_OPS_EVT], [0, 0])
            self.dw = self.red_dwb[:self.weights.size].reshape(self.weights.shape)
            self.db = self.red_dwb[-self.bias.size:].reshape(self.bias.shape)

class Input(Layer):

    def __init__(self, shape=(1,)):
        super().__init__(shape)

class FC(Layer):

    def __init__(self, shape=(1,), activation=None, 
                 weights_initializer="glorot_initializer",
                 bias_initializer="zeros_initializer"):
        super().__init__(shape)
        self.act = activation
        self.weights_initializer = getattr(NN_util, weights_initializer)
        self.bias_initializer = getattr(NN_util, bias_initializer)
        
    def initialize(self):
        self.weights = self.weights_initializer((np.prod(self.prev_layer.shape), np.prod(self.shape[0])), self)
        self.bias = self.bias_initializer((np.prod(self.shape),), self)
        self.params = np.prod(self.weights.shape) + np.prod(self.bias.shape)
        
    def show(self):
        super().show("│{:^17s}│{:^9s}│{:^9s}│".format(str(self.weights.shape),"",""))

    def forward(self, prev_a):
        self.tracer.emit_event(PYDL_OPS_EVT, self.id * PYDL_OPS_NUM_EVTS + 1)
        res = self.matmul(prev_a.reshape(prev_a.shape[0], -1), self.weights)
        self.tracer.emit_event(PYDL_OPS_EVT, 0)
        self.a = res + self.bias
        
    def backward(self, prev_dx):
        self.tracer.emit_event(PYDL_OPS_EVT, self.id * PYDL_OPS_NUM_EVTS + 3)
        dx = self.matmul(prev_dx, self.weights.T)
        self.tracer.emit_event(PYDL_OPS_EVT, 0)

        self.tracer.emit_event(PYDL_OPS_EVT, self.id * PYDL_OPS_NUM_EVTS + 5)
        self.dw = self.matmul(self.prev_layer.a.reshape(self.prev_layer.a.shape[0], -1).T, prev_dx)
        self.tracer.emit_event(PYDL_OPS_EVT, 0)
        self.db = prev_dx.sum(axis=0)
        return dx
        
class Conv2D(Layer):

    def __init__(self, nfilters=1, filter_shape=(3, 3), padding=0, stride=1, 
                 activation=None, weights_initializer="glorot_initializer",
                 bias_initializer="zeros_initializer"):
        super().__init__()
        self.co = nfilters
        self.filter_shape = filter_shape
        self.padding = padding
        self.stride = stride
        self.weights_initializer = getattr(NN_util, weights_initializer)
        self.bias_initializer = getattr(NN_util, bias_initializer)
        self.act = activation

    def initialize(self):
        self.ci, self.hi, self.wi = self.prev_layer.shape
        self.kh, self.kw = self.filter_shape

        self.weights = self.weights_initializer(((self.co,)+(self.ci,)+self.filter_shape), self)
        self.bias = self.bias_initializer((self.co,), self)

        self.ho = floor((self.hi + 2 * self.padding - self.kh) / self.stride) + 1
        self.wo = floor((self.wi + 2 * self.padding - self.kw) / self.stride) + 1
        self.shape = (self.co, self.ho, self.wo)
        self.params = np.prod(self.weights.shape) + np.prod(self.bias.shape)

    def show(self):
        super().show("│{:^17s}│{:^9d}│{:^9d}│".format(str(self.weights.shape), self.padding, self.stride))

    def forward(self, prev_a):
        self.tracer.emit_event(PYDL_OPS_EVT, self.id * PYDL_OPS_NUM_EVTS + 2)
        self.prev_a_cols = im2col_cython(prev_a, self.kh, self.kw, self.padding, self.stride)
        self.tracer.emit_event(PYDL_OPS_EVT, 0)

        w_cols = self.weights.reshape(self.co, -1)
        self.tracer.emit_event(PYDL_OPS_EVT, self.id * PYDL_OPS_NUM_EVTS + 1)
        res = self.matmul(w_cols, self.prev_a_cols)
        self.tracer.emit_event(PYDL_OPS_EVT, 0)

        a = (res.T + self.bias).T
        self.a = a.reshape(self.co, -1, self.ho, self.wo).transpose(1, 0, 2, 3)
        #printf("forward:", type(self).__name__, self.shape, self.a.shape)

    def backward(self, prev_dx):      
        #printf(" _%d_%s_get_gradient:" % (self.id, type(self).__name__), self.shape)
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

class Pool2D(Layer):

    def __init__(self, pool_shape=(2,2), func='max', padding=0, stride=1):
        super().__init__()
        self.pool_shape = pool_shape
        self.func_str = func
        self.padding = padding
        self.stride = stride
        if func != "max":
            raise func + "is not yet supported!"        

    def initialize(self):
        self.ci, self.hi, self.wi = self.prev_layer.shape
        self.kh, self.kw = self.pool_shape
        self.ho = floor((self.hi - self.kh) / self.stride) + 1
        self.wo = floor((self.wi - self.kw) / self.stride) + 1
        self.co = self.ci
        self.shape = (self.co, self.ho, self.wo)
        self.n = np.prod(self.shape)

    def show(self):
        super().show("│{:^17s}│{:^9d}│{:^9d}│".format("", self.padding, self.stride))

    def forward(self, prev_a):
        prev_a_ = prev_a.reshape(prev_a.shape[0] * self.ci, 1, self.hi, self.wi)
        self.tracer.emit_event(PYDL_OPS_EVT, self.id * PYDL_OPS_NUM_EVTS + 2)
        a_cols = im2col_cython(prev_a_, self.kh, self.kw, self.padding, self.stride)
        self.tracer.emit_event(PYDL_OPS_EVT, 0)

        self.maxids = tuple([a_cols.argmax(axis=0), np.arange(a_cols.shape[1])])
        self.a = a_cols[self.maxids].reshape(prev_a.shape[0], self.co, self.ho, self.wo)        
        #printf("forward:", type(self).__name__, self.shape, self.a.shape)

    def backward(self, prev_dx):     
        #printf(" _%d_%s_get_gradient:" % (self.id, type(self).__name__), self.shape)
        dx_cols = np.zeros((self.kh * self.kw, np.prod(prev_dx.shape))).astype(self.dtype)
        dx_cols[self.maxids] = prev_dx.flatten()
        self.tracer.emit_event(PYDL_OPS_EVT, self.id * PYDL_OPS_NUM_EVTS + 4)
        dx = col2im_cython(dx_cols, prev_dx.shape[0] * self.ci, 1, self.hi, self.wi, 
                           self.kh, self.kw, self.padding, self.stride)
        self.tracer.emit_event(PYDL_OPS_EVT, 0)
        dx = dx.reshape(prev_dx.shape[0], self.ci, self.hi, self.wi)
        return dx

class Dropout(Layer):

    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def initialize(self):
        self.shape = self.prev_layer.shape
        
    def forward(self, prev_a):
        self.mask = np.random.binomial(1, self.prob, size=self.shape).astype(self.dtype) / self.prob
        self.a = prev_a * self.mask

    def backward(self, prev_dx):
        return prev_dx * self.mask
 
# Flatten layers are not needed anymore!
# class Flatten(Layer):
# 
#     def __init__(self):
#         super().__init__()
# 
#     def initialize(self):
#         self.shape = (np.prod(self.prev_layer.shape),)
#         self.n = np.prod(self.shape)
# 
#     def forward(self, prev_a):
#         self.a = prev_a.reshape(prev_a.shape[0], -1)
# 
#     def backward(self, prev_dx):
#         return prev_dx.reshape((prev_dx.shape[0],) + self.prev_layer.shape)
