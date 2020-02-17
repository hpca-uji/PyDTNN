import numpy as np
import math
import random
import NN_utils

#from mpi4py import MPI

from NN_utils import printf, im2col, col2im, dilate_and_pad, PYDL_EVT, PYDL_OPS_EVT, PYDL_NUM_EVTS, PYDL_OPS_EVT, PYDL_OPS_NUM_EVTS
from math import floor

class Layer():
    """ Layer of a neural network """

    def __init__(self, shape=()):
        self.shape = shape
        self.n = np.prod(shape)
        self.prev_layer = None
        self.next_layer = None
        self.id = None
        self.changeW = []

    def show(self):
        print('Layer    ', self.id)
        print('Type     ', type(self).__name__)
        print('#Neurons ', self.n)
        print('Shape    ', self.shape)

    def backward(self, dX= []):
        printf(" _%d_%s_backward: " % (self.id, type(self).__name__), self.shape, len(dX))
        if dX == []: self.dx = self.next_layer.get_gradient()
        else:        self.dx = dX
        #printf("    ", type(self).__name__, " self.dx", self.dx.shape)

    def reduce_weights(self, comm):
        if comm != None and len(self.changeW) > 0:
           self.WB = np.append(self.changeW.reshape(-1), self.changeB.reshape(-1))
           self.red_WB = np.zeros_like(self.WB)
           self.req_AR = comm.Iallreduce( self.WB, self.red_WB, op = MPI.SUM )
     
    def wait_allreduce(self, comm):
        if comm != None and len(self.changeW) > 0:
           self.req_AR.Wait()
           self.changeW = self.red_WB[0:self.weights.size].reshape(self.weights.shape)
           self.changeB = self.red_WB[self.WB.size-self.bias.size:].reshape(self.bias.shape)    
     
    def reduce_weights_sync(self, comm):
        if comm != None and len(self.changeW) > 0:
           self.WB = np.append(self.changeW.reshape(-1), self.changeB.reshape(-1))
           self.red_WB = np.zeros_like(self.WB)
           self.model.emit_nevent([PYDL_EVT, PYDL_OPS_EVT], [self.id * PYDL_NUM_EVTS + 5, self.id * PYDL_OPS_NUM_EVTS + 9])
           comm.Allreduce( self.WB, self.red_WB, op = MPI.SUM )
           self.model.emit_nevent([PYDL_EVT, PYDL_OPS_EVT], [0, 0])
           self.changeW = self.red_WB[0:self.weights.size].reshape(self.weights.shape)
           self.changeB = self.red_WB[self.WB.size-self.bias.size:].reshape(self.bias.shape)    

class Input(Layer):
    """ Input layer for neural network """

    def __init__(self, shape=(1,)):
        super().__init__(shape)

class FC(Layer):
    """ FC layer for neural network """

    def __init__(self, shape=(1,), activation="sigmoid"):
        super().__init__(shape)
        self.act= getattr(NN_utils, activation)
        self.act_der= getattr(NN_utils, "%s_derivate" % activation) 
        self.mp = 1 
        
    def initialize(self):
        self.weights = np.random.uniform(-1, 1, (self.n, self.prev_layer.n))
        self.bias = np.random.uniform(-1, 1, (self.n, 1))

    def infer(self, prev_a):
        self.model.emit_event(PYDL_OPS_EVT, self.id * PYDL_OPS_NUM_EVTS + 2)
        res_matmul = NN_utils.matmul(self.weights, prev_a, 
            "%d_%s_inference_matmul" % (self.id, type(self).__name__))
        self.model.emit_event(PYDL_OPS_EVT, 0)

        a = self.act(res_matmul + self.bias)
        return a

    def forward(self, prev_a):        
        self.model.emit_event(PYDL_OPS_EVT, self.id * PYDL_OPS_NUM_EVTS + 4)
        res_matmul = NN_utils.matmul(self.weights, prev_a,
            "%d_%s_forward_matmul" % (self.id, type(self).__name__))
        self.model.emit_event(PYDL_OPS_EVT, 0)

        z = res_matmul + self.bias
        self.a = self.act(z)
        self.dz = self.act_der(z)
        printf("forward:", type(self).__name__, self.shape, self.a.shape)
        
    def get_gradient(self):
        printf(" _%d_%s_get_gradient:" % (self.id, type(self).__name__), self.shape)

        self.model.emit_event(PYDL_OPS_EVT, (self.id-1) * PYDL_OPS_NUM_EVTS + 6)
        res_matmul = NN_utils.matmul(self.weights.T, self.dx,
            "%d_%s_compute_dX_matmul" % (self.id-1, type(self).__name__))
        self.model.emit_event(PYDL_OPS_EVT, 0)

        dX = res_matmul * self.prev_layer.dz
        return dX

    def calculate_change(self, b):
        self.model.emit_event(PYDL_OPS_EVT, self.id * PYDL_OPS_NUM_EVTS + 8)
        self.changeW = NN_utils.matmul(self.dx, self.prev_layer.a.T,
            "%d_%s_compute_dW_matmul" % (self.id, type(self).__name__))
        self.model.emit_event(PYDL_OPS_EVT, 0)

        self.changeB = self.dx.sum(axis=1).reshape(self.bias.shape[0], 1)

    def update_weights(self, eta, b):
        self.weights-= eta * self.changeW
        self.bias-= eta * self.changeB
        
class Conv2D(Layer):
    """ Conv2D layer for neural network """

    def __init__(self, nfilters=1, filter_shape=(3, 3, 1), padding=0, stride=1, activation="sigmoid"):
        super().__init__()
        self.co = nfilters
        self.filter_shape = filter_shape
        self.padding = padding
        self.stride = stride
        self.act= getattr(NN_utils, activation)
        self.act_der= getattr(NN_utils, "%s_derivate" % activation)  
        self.mp = 1         
        self.cached_idx_fp = self.cached_idx_gc = self.cached_idx_wu = self.b = None      

    def initialize(self):
        self.weights = np.random.uniform(-1, 1, (self.co,)+self.filter_shape)
        self.bias = np.random.uniform(-1, 1, (self.co,))
        self.hi, self.wi, self.ci = self.prev_layer.shape
        self.kh, self.kw, self.ci = self.filter_shape
        self.ho = floor((self.hi + 2 * self.padding - self.kh) / self.stride) + 1
        self.wo = floor((self.wi + 2 * self.padding - self.kw) / self.stride) + 1
        self.shape = (self.ho, self.wo, self.co)
        self.n = np.prod(self.shape)

    def show(self):
        super().show()
        print('Padding  ', self.padding)
        print('Stride   ', self.stride) 
        print('#Filters ', self.weights.shape)

    def infer(self, prev_a):
        prev_a = dilate_and_pad(prev_a.transpose(3, 2, 0, 1), self.padding)

        self.model.emit_event(PYDL_OPS_EVT, self.id * PYDL_OPS_NUM_EVTS + 1)
        patched_act, self.cached_idx_fp= im2col(prev_a, self.kh, self.kw, 
            self.ci, self.ho, self.wo, self.stride, self.cached_idx_fp,
            "%d_%s_inference_im2col" % (self.id, type(self).__name__))
        self.model.emit_event(PYDL_OPS_EVT, 0)

        patched_weights= self.weights.transpose(0, 3, 1, 2).reshape(self.co, -1)

        self.model.emit_event(PYDL_OPS_EVT, self.id * PYDL_OPS_NUM_EVTS + 2)
        res_matmul = NN_utils.matmul(patched_weights, patched_act,
            "%d_%s_inference_matmul" % (self.id, type(self).__name__))
        self.model.emit_event(PYDL_OPS_EVT, 0)

        z = ((res_matmul.T + self.bias).T)
        z = z.reshape(self.co, self.ho, self.wo, -1).transpose(1, 2, 0, 3) # PyNN format
        a = self.act(z)
        return a

    def forward(self, prev_a):
        prev_a = dilate_and_pad(prev_a.transpose(3, 2, 0, 1), self.padding)

        self.model.emit_event(PYDL_OPS_EVT, self.id * PYDL_OPS_NUM_EVTS + 3)
        patched_act, self.cached_idx_fp= im2col(prev_a, self.kh, self.kw, 
            self.ci, self.ho, self.wo, self.stride, self.cached_idx_fp,
            "%d_%s_forward_im2col" % (self.id, type(self).__name__))
        self.model.emit_event(PYDL_OPS_EVT, 0)

        patched_weights= self.weights.transpose(0, 3, 1, 2).reshape(self.co, -1)

        self.model.emit_event(PYDL_OPS_EVT, self.id * PYDL_OPS_NUM_EVTS + 4)
        res_matmul = NN_utils.matmul(patched_weights, patched_act,
            "%d_%s_forward_matmul" % (self.id, type(self).__name__))
        self.model.emit_event(PYDL_OPS_EVT, 0)

        z = ((res_matmul.T + self.bias).T)
        z = z.reshape(self.co, self.ho, self.wo, -1).transpose(1, 2, 0, 3) # PyNN format
        self.a = self.act(z)
        self.dz = self.act_der(z)        
        printf("forward:", type(self).__name__, self.shape, self.a.shape)

    def get_gradient(self):
        printf(" _%d_%s_get_gradient:" % (self.id, type(self).__name__), self.shape)
        d = dilate_and_pad(self.dx.transpose(3, 2, 0, 1), self.kh-self.padding-1, self.stride)

        self.model.emit_event(PYDL_OPS_EVT, (self.id-1) * PYDL_OPS_NUM_EVTS + 5)
        patched_matrix, self.cached_idx_gc= im2col(d, self.kh, self.kw, 
            self.co, self.hi, self.wi, 1, self.cached_idx_gc,
            "%d_%s_compute_dX_im2col" % (self.id-1, type(self).__name__))
        self.model.emit_event(PYDL_OPS_EVT, 0)

        patched_weights= np.rot90(self.weights.transpose(3, 0, 1, 2), 2, axes=(2,3)).reshape(self.ci, -1)

        self.model.emit_event(PYDL_OPS_EVT, (self.id-1) * PYDL_OPS_NUM_EVTS + 6)
        res_matmul = NN_utils.matmul(patched_weights, patched_matrix,
            "%d_%s_compute_dX_matmul" % (self.id-1, type(self).__name__))
        self.model.emit_event(PYDL_OPS_EVT, 0)

        dx = res_matmul.reshape(self.ci, self.hi, self.wi, -1).transpose(1, 2, 0, 3) # PyNN format
        dx*= self.prev_layer.dz
        return dx

    def calculate_change(self, b):
        d = dilate_and_pad(self.dx.transpose(3, 2, 0, 1), 0, self.stride)
        act = dilate_and_pad(self.prev_layer.a.transpose(2, 3, 0, 1), self.padding)        
        if self.b != b: self.cached_idx_wu = None
        self.b, co, ho, wo = d.shape

        self.model.emit_event(PYDL_OPS_EVT, self.id * PYDL_OPS_NUM_EVTS + 7)
        patched_act, self.cached_idx_wu= im2col(act, ho, wo, 
            b, self.kh, self.kw, 1, self.cached_idx_wu,
            "%d_%s_compute_dW_im2col" % (self.id, type(self).__name__))
        self.model.emit_event(PYDL_OPS_EVT, 0)

        patched_grad= d.transpose(1, 0, 2, 3).reshape(self.co, -1)

        self.model.emit_event(PYDL_OPS_EVT, self.id * PYDL_OPS_NUM_EVTS + 8)
        res_matmul = NN_utils.matmul(patched_grad, patched_act,
            "%d_%s_compute_dW_matmul" % (self.id, type(self).__name__))
        self.model.emit_event(PYDL_OPS_EVT, 0)

        self.changeW = res_matmul.reshape(self.co, self.kh, self.kw, self.ci)
        self.changeB = self.dx.transpose(2, 3, 0, 1).sum(axis=(1,2,3))

    def update_weights(self, eta, b):
        self.weights-= eta * self.changeW
        self.bias   -= eta * self.changeB

class Pool2D(Layer):
    """ Pool2D layer for neural network """

    def __init__(self, pool_shape=(2,2), func='max', stride=1):
        super().__init__()
        self.pool_shape = pool_shape
        self.func_str = func
        self.stride = stride
        self.cached_idx = None         

    def initialize(self,):
        self.hi, self.wi, self.ci = self.prev_layer.shape
        self.kh, self.kw = self.pool_shape
        self.stride = self.kh
        self.ho = floor((self.hi - self.kh) / self.stride) + 1
        self.wo = floor((self.wi - self.kw) / self.stride) + 1
        self.co = self.ci
        self.hp, self.wp = (self.hi - self.kh) % self.stride, (self.wi - self.kw) % self.stride
        self.shape = (self.ho, self.wo, self.co)
        self.n = np.prod(self.shape)

    def infer(self, prev_a):
        b = prev_a.shape[-1]
        prev_a = prev_a.transpose(3, 2, 0, 1)[...,:self.hi-self.hp,:self.wi-self.wp]
        prev_a_ = prev_a.reshape(b * self.ci, 1, self.hi-self.hp, self.wi-self.wp)

        self.model.emit_event(PYDL_OPS_EVT, self.id * PYDL_OPS_NUM_EVTS + 1)
        patched_a, self.cached_idx= im2col(prev_a_, self.kh, self.kw, 
            1, self.ho, self.wo, self.stride, self.cached_idx,
            "%d_%s_inference_im2col" % (self.id, type(self).__name__))
        self.model.emit_event(PYDL_OPS_EVT, 0)

        if   self.func_str == "max": a = patched_a.max(axis=0)
        elif self.func_str == "avg": a = patched_a.mean(axis=0)            
        a = a.reshape(self.ho, self.wo, b, self.co).transpose(0, 1, 3, 2) # PyNN format
        return a

    def forward(self, prev_a):

        b = prev_a.shape[-1]
        prev_a = prev_a.transpose(3, 2, 0, 1)[...,:self.hi-self.hp,:self.wi-self.wp]
        prev_a_ = prev_a.reshape(b * self.ci, 1, self.hi-self.hp, self.wi-self.wp)

        self.model.emit_event(PYDL_OPS_EVT, self.id * PYDL_OPS_NUM_EVTS + 3)
        patched_a,  self.cached_idx= im2col(prev_a_, self.kh, self.kw, 
            1, self.ho, self.wo, self.stride, self.cached_idx,
            "%d_%s_forward_im2col" % (self.id, type(self).__name__))
        self.model.emit_event(PYDL_OPS_EVT, 0)

        if self.func_str == "max":
            self.a = patched_a.max(axis=0).reshape(self.ho, self.wo, b, self.co).transpose(0, 1, 3, 2) # PyNN format
            self.argmax = self.tuple([patched_a.argmax(axis=0), np.arange(patched_a.shape[1])])

            prev_dz = self.prev_layer.dz.transpose(3, 2, 0, 1)[...,:self.hi-self.hp,:self.wi-self.wp]
            prev_dz = prev_dz.reshape(b * self.ci, 1, self.hi-self.hp, self.wi-self.wp)

            self.model.emit_event(PYDL_OPS_EVT, self.id * PYDL_OPS_NUM_EVTS + 3)
            patched_dz, self.cached_idx= im2col(prev_dz, self.kh, self.kw, 
                1, self.ho, self.wo, self.stride, self.cached_idx,
                "%d_%s_forward2_im2col" % (self.id, type(self).__name__))
            self.model.emit_event(PYDL_OPS_EVT, 0)

            self.dz = patched_dz[self.argmax].reshape(self.ho, self.wo, b, self.co).transpose(0, 1, 3, 2) # PyNN format


        elif self.func_str == "avg":
            self.a = patched_a.mean(axis=0).reshape(self.ho, self.wo, b, self.co).transpose(0, 1, 3, 2) # PyNN format
            prev_dz = self.prev_layer.dz.transpose(3, 2, 0, 1)[...,:self.hi-self.hp,:self.wi-self.wp]
            prev_dz = prev_dz.reshape(b * self.ci, 1, self.hi-self.hp, self.wi-self.wp)

            self.model.emit_event(PYDL_OPS_EVT, self.id * PYDL_OPS_NUM_EVTS + 3)
            patched_dz, self.cached_idx= im2col(prev_dz, self.kh, self.kw, 
                1, self.ho, self.wo, self.stride, self.cached_idx,
                "%d_%s_forward2_im2col" % (self.id, type(self).__name__))
            self.model.emit_event(PYDL_OPS_EVT, 0)

            self.dz = patched_dz.mean(axis=0).reshape(self.ho, self.wo, b, self.co).transpose(0, 1, 3, 2) # PyNN format

    def get_gradient(self):     
        printf(" _%d_%s_get_gradient:" % (self.id, type(self).__name__), self.shape)
        if self.func_str == "max":
                            # Expected (h, w, b, c)    PyNN to requested  o.transpose(0,1,3,2) , Normal to requested o.transpose(2,3,0,1)
            b = self.dx.shape[-1]
            dx = np.repeat(self.dx.transpose(0,1,3,2).reshape(1,-1), self.kh*self.kw, axis=0)
            patched_dx = np.zeros_like(dx)
            patched_dx[self.argmax] = dx[self.argmax]
            dx, self.cached_idx = col2im(patched_dx, (b * self.ci, 1, self.hi-self.hp, self.wi-self.wp), \
                                     self.kh, self.kw, self.ho, self.wo, self.stride, self.cached_idx)
            dx = dx.reshape(b, self.ci, self.hi-self.hp, self.wi-self.wp)

        elif self.func_str == "avg":
            dx = np.kron(self.dx.transpose(3, 2, 0, 1), (np.ones((self.kh, self.kw)) / (self.kh * self.kw)) )
        dx = np.pad(dx, ((0, 0), (0, 0), (0, self.hp), (0, self.wp)), mode='constant').transpose(2, 3, 1, 0) # PyNN format
        return dx
                     
    def calculate_change(self, b):
        pass

    def update_weights(self, eta, b):
        pass

class Flatten(Layer):
    """ Flatten layer for neural network """

    def __init__(self):
        super().__init__()

    def initialize(self):
        self.shape = (np.prod(self.prev_layer.shape),)
        self.n = np.prod(self.shape)

    def infer(self, prev_a):
        b = prev_a.shape[-1]
        a = prev_a.reshape(-1, b)
        return a

    def forward(self, prev_a):
        b = prev_a.shape[-1]
        self.a = prev_a.reshape(-1, b)
        if hasattr(self.prev_layer, 'dz'):
            self.dz = self.prev_layer.dz.reshape(-1, b)
        else:
            self.dz = np.ones(self.a.shape)
        printf("forward:", type(self).__name__, self.shape, self.a.shape)

    def get_gradient(self):
        printf("  get_gradient:", type(self).__name__, self.shape)
        b = self.a.shape[-1]
        dX = self.dx.reshape(self.prev_layer.shape + (b,))
        return dX

    def calculate_change(self, b):
        pass

    def update_weights(self, eta, b):
        pass

class Dropout(Layer):
    """ Dropout layer for neural network """

    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def initialize(self):
        self.shape = self.prev_layer.shape
        self.n = np.prod(self.shape)

    def show(self):
        super().show()
        print('Prob:    ', self.prob)

    def infer(self, prev_a):
        mask = np.random.binomial(1, self.prob, size=prev_a.shape) / self.prob
        return prev_a * mask

    def forward(self, prev_a):
        self.mask = np.random.binomial(1, self.prob, size=prev_a.shape) / self.prob
        self.a = prev_a * self.mask        
        if hasattr(self.prev_layer, 'dz'):
            self.dz = self.prev_layer.dz * self.mask 
        else:
            self.dz = np.ones(self.a.shape) * self.mask 
        printf("forward:", type(self).__name__, self.shape, self.a.shape)

    def get_gradient(self):
        printf(" _%d_%s_get_gradient:" % (self.id, type(self).__name__), self.shape)
        return self.dx * self.mask

    def calculate_change(self, b):
        pass

    def update_weights(self, eta, b):
        pass
