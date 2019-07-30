import numpy as np
import math
import random
import NN_utils
from NN_utils import printf, im2col, col2im, dilate_and_pad
from math import floor

class Layer():
    """ Layer of a neural network """

    def __init__(self, shape=()):
        self.shape = shape
        self.n = np.prod(shape)
        self.prev_layer = None
        self.next_layer = None

    def show(self):
        print('Layer    ')
        print('Type     ', type(self).__name__)
        print('#Neurons ', self.n)
        print('Shape    ', self.shape)

    def backward(self, dX= []):
        printf("\nbackward:", type(self).__name__, self.shape)
        if dX == []: self.dx = self.next_layer.get_gradient()
        else:        self.dx = dX
        printf("    ", type(self).__name__, " self.dx", self.dx.shape)

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
        
    def initialize(self):
        self.weights = np.random.uniform(-1, 1, (self.n, self.prev_layer.n))
        self.bias = np.random.uniform(-1, 1, (self.n, 1))

    def infer(self, prev_a):  
        z = self.weights @ prev_a + self.bias
        return self.act(z)

    def forward(self, prev_a):        
        z = self.weights @ prev_a + self.bias
        self.a = self.act(z)
        self.dz = self.act_der(z)
        printf("forward:", type(self).__name__, self.shape, self.a.shape)
        
    def get_gradient(self):
        printf("  get_gradient:", type(self).__name__, self.shape)
        return (self.weights.T @ self.dx) * self.prev_layer.dz

    def update_weights(self, eta, b):
        self.weights-= (eta/b) * (self.dx @ self.prev_layer.a.T)
        self.bias-= (eta/b) * self.dx.sum(axis=1).reshape(self.bias.shape[0], 1)
        
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
        #z = NN_utils.convolve_scipy(prev_a, self.weights, self.bias, self.padding, self.stride)
        #z = NN_utils.convolve(prev_a, self.weights, self.bias, self.padding, self.stride) 
        prev_a = dilate_and_pad(prev_a.transpose(3, 2, 0, 1), self.padding)
        patched_act, self.cached_idx_fp= im2col(prev_a, self.kh, self.kw, self.ci, self.ho, self.wo, self.stride, self.cached_idx_fp)
        patched_weights= self.weights.transpose(0, 3, 1, 2).reshape(self.co, -1)
        z = (((patched_weights @ patched_act).T + self.bias).T)
        z = z.reshape(self.co, self.ho, self.wo, -1).transpose(1, 2, 0, 3) # PyNN format
        return self.act(z)

    def forward(self, prev_a):
        prev_a = dilate_and_pad(prev_a.transpose(3, 2, 0, 1), self.padding)
        patched_act, self.cached_idx_fp= im2col(prev_a, self.kh, self.kw, self.ci, self.ho, self.wo, self.stride, self.cached_idx_fp)
        patched_weights= self.weights.transpose(0, 3, 1, 2).reshape(self.co, -1)
        z = (((patched_weights @ patched_act).T + self.bias).T)
        z = z.reshape(self.co, self.ho, self.wo, -1).transpose(1, 2, 0, 3) # PyNN format
        self.a  = self.act(z)
        self.dz = self.act_der(z)        
        printf("forward:", type(self).__name__, self.shape, self.a.shape)

    def get_gradient(self):
        printf("  get_gradient:", type(self).__name__, self.shape)
        d = dilate_and_pad(self.dx.transpose(3, 2, 0, 1), self.kh-self.padding-1, self.stride)
        patched_matrix, self.cached_idx_gc= im2col(d, self.kh, self.kw, self.co, self.hi, self.wi, 1, self.cached_idx_gc)
        patched_weights= np.rot90(self.weights.transpose(3, 0, 1, 2), 2, axes=(2,3)).reshape(self.ci, -1)
        dx = (patched_weights @ patched_matrix).reshape(self.ci, self.hi, self.wi, -1).transpose(1, 2, 0, 3) # PyNN format
        dx*= self.prev_layer.dz
        return dx
        # Old code left as reference
        # for b_ in range(b):
        #     for ci_ in range(ci):
        #         for co_ in range(co):
        #             dx[...,ci_,b_] += convolve2d(self.dx[...,co_,b_], self.weights[co_,...,ci_], mode='full')
        #         dx[...,ci_,b_] *= self.prev_layer.dz[...,ci_,b_]    

    def update_weights(self, eta, b):
        d = dilate_and_pad(self.dx.transpose(3, 2, 0, 1), 0, self.stride)
        act = dilate_and_pad(self.prev_layer.a.transpose(2, 3, 0, 1), self.padding)        
        if self.b != b: self.cached_idx_wu = None
        self.b, co, ho, wo = d.shape
        patched_act, self.cached_idx_wu= im2col(act, ho, wo, b, self.kh, self.kw, 1, self.cached_idx_wu)
        patched_grad= d.transpose(1, 0, 2, 3).reshape(self.co, -1)
        dw = (patched_grad @ patched_act).reshape(self.co, self.kh, self.kw, self.ci)
        self.weights-= (eta/b) * dw
        self.bias   -= (eta/b) * self.dx.transpose(2, 3, 0, 1).sum(axis=(1,2,3))
        # Old code left as reference
        # for b_ in range(b):
        #     for co_ in range(co):
        #         for ci_ in range(ci):
        #             self.weights[co_,...,ci_] -= (eta/b) * \
        #                convolve2d(self.prev_layer.a[...,ci_,b_], self.dx[...,co_,b_], mode='valid')
        #         self.bias[co_] -= (eta/b) * self.dx[...,co_,b_].sum(axis=-1).sum(axis=-1)        

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
        prev_a = prev_a.transpose(3, 2, 0, 1).reshape(b * self.ci, 1, self.hi, self.wi)
        patched_z, self.cached_idx= im2col(prev_a, self.kh, self.kw, 1, self.ho, self.wo, self.stride, self.cached_idx)
        if   self.func_str == "max": z = patched_z.max(axis=0)
        elif self.func_str == "avg": z = patched_z.mean(axis=0)            
        z = z.reshape(self.ho, self.wo, b, self.co).transpose(0, 1, 3, 2) # PyNN format
        return z

    def forward(self, prev_a):
        b = prev_a.shape[-1]
        prev_a = prev_a.transpose(3, 2, 0, 1)
        prev_a_ = prev_a.reshape(b * self.ci, 1, self.hi, self.wi)[...,:self.hi-self.hp,:self.wi-self.wp]
        patched_a,  self.cached_idx= im2col(prev_a_,  self.kh, self.kw, 1, self.hi-self.hp, self.wi-self.wp, self.stride, self.cached_idx)        

        if self.func_str == "max":
            self.a = patched_a.max(axis=0).reshape(self.ho, self.wo, b, self.co).transpose(0, 1, 3, 2) # PyNN format
            r = np.kron(self.a.transpose(3, 2, 0, 1), np.ones(self.pool_shape))
            self.mask = np.equal(prev_a[...,:self.hi-self.hp,:self.wi-self.wp], r).astype(int)
            prev_dz = self.prev_layer.dz.transpose(3, 2, 0, 1)[...,:self.hi-self.hp,:self.wi-self.wp] * self.mask
            prev_dz = prev_dz.reshape(b * self.ci, 1, self.hi-self.hp, self.wi-self.wp)
            patched_dz, self.cached_idx= im2col(prev_dz, self.kh, self.kw, 1, self.ho, self.wo, self.stride, self.cached_idx)
            self.dz = patched_dz.max(axis=0).reshape(self.ho, self.wo, b, self.co).transpose(0, 1, 3, 2) # PyNN format

        elif self.func_str == "avg":
            self.a = patched_a.mean(axis=0).reshape(self.ho, self.wo, b, self.co).transpose(0, 1, 3, 2) # PyNN format
            prev_dz = self.prev_layer.dz.transpose(3, 2, 0, 1)[...,:self.hi-self.hp,:self.wi-self.wp]
            prev_dz = prev_dz.reshape(b * self.ci, 1, self.hi-self.hp, self.wi-self.wp)
            patched_dz, self.cached_idx= im2col(prev_dz, self.kh, self.kw, 1, self.hi-self.hp, self.wi-self.wp, self.stride, self.cached_idx)
            self.dz = patched_dz.mean(axis=0).reshape(self.ho, self.wo, b, self.co).transpose(0, 1, 3, 2) # PyNN format

    def get_gradient(self):     
        if self.func_str == "max":
            dx = np.kron(self.dx.transpose(3, 2, 0, 1), np.ones(self.pool_shape)) * self.mask
        elif self.func_str == "avg":
            dx = np.kron(self.dx.transpose(3, 2, 0, 1), (np.ones((self.kh, self.kw)) / (self.kh * self.kw)) )
        dx = np.pad(dx, ((0, 0), (0, 0), (0, self.hp), (0, self.wp)), mode='constant').transpose(2, 3, 1, 0) # PyNN format
        return dx

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
        return prev_a.reshape(-1, b)

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
        return self.dx.reshape(self.prev_layer.shape + (b,))

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
        printf("  get_gradient:", type(self).__name__, self.shape)
        return self.dx * self.mask

    def update_weights(self, eta, b):
        pass
