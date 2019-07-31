import numpy as np
import math
import random
import NN_utils
from NN_utils import printf, im2col_indices
from math import ceil


from scipy.signal import convolve2d
from skimage.measure import block_reduce

class Layer():
    """ Layer of a neural network """

    def __init__(self, shape=()):
        self.shape = shape
        self.n = np.prod(shape)
        self.prev_layer = None
        self.next_layer = None
        self.changeW = []

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
        self.mp = 1 
        
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

    def calculate_change(self):
        self.changeW = self.dx @ self.prev_layer.a.T
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

    def initialize(self):
        self.weights = np.random.uniform(-1, 1, (self.co,)+self.filter_shape)
        self.bias = np.random.uniform(-1, 1, (self.co,))
        self.hi, self.wi, self.ci = self.prev_layer.shape
        self.kh, self.kw, self.ci = self.filter_shape
        self.ho = ceil((self.hi + 2 * self.padding - self.kh) / self.stride + 1)
        self.wo = ceil((self.wi + 2 * self.padding - self.kw) / self.stride + 1)
        self.shape = (self.ho, self.wo, self.co)
        self.n = np.prod(self.shape)

    def show(self):
        super().show()
        print('#Filters ', self.weights.shape)

    def infer(self, prev_a):
        #z = NN_utils.convolve_scipy(prev_a, self.weights, self.bias, self.padding, self.stride)
        #z = NN_utils.convolve(prev_a, self.weights, self.bias, self.padding, self.stride) 
        p= self.padding
        prev_a = prev_a.transpose(3, 2, 0, 1) # b, c, h, w, this is needed for padding
        prev_a = np.pad(prev_a, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
        patched_act= im2col_indices(prev_a, self.kh, self.kw, self.ci, self.ho, self.wo, self.stride)
        #patched_act= im2col_indices(prev_a, self.kh, self.kw, self.ci, self.ho, self.wo, self.padding, self.stride)
        patched_weights= self.weights.transpose(0, 3, 1, 2).reshape(self.co, -1)
        z = (((patched_weights @ patched_act).T + self.bias).T)
        z = z.reshape(self.co, self.ho, self.wo, -1).transpose(1, 2, 0, 3) # PyNN format
        return self.act(z)

    def forward(self, prev_a):
        p= self.padding
        prev_a = prev_a.transpose(3, 2, 0, 1) # b, c, h, w, this is needed for padding
        prev_a = np.pad(prev_a, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
        patched_act= im2col_indices(prev_a, self.kh, self.kw, self.ci, self.ho, self.wo, self.stride)       
        #patched_act= im2col_indices(prev_a, self.kh, self.kw, self.ci, self.ho, self.wo, self.padding, self.stride)
        patched_weights= self.weights.transpose(0, 3, 1, 2).reshape(self.co, -1)
        z = (((patched_weights @ patched_act).T + self.bias).T)
        z = z.reshape(self.co, self.ho, self.wo, -1).transpose(1, 2, 0, 3) # PyNN format
        self.a = self.act(z)
        self.dz = self.act_der(z)        
        printf("forward:", type(self).__name__, self.shape, self.a.shape)

    def get_gradient(self):
        printf("  get_gradient:", type(self).__name__, self.shape)
        p = self.kh-self.padding-1
        d = self.dx.transpose(3, 2, 0, 1) # b, c, h, w
        if self.stride > 1: 
            mask = np.zeros((self.stride, self.stride));  mask[0,0] = 1
            d = np.kron(d, mask)[...,:-(self.stride-1),:-(self.stride-1)]
        d = np.pad(d, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
        patched_matrix= im2col_indices(d, self.kh, self.kw, self.co, self.hi, self.wi, 1)
        #patched_matrix= im2col_indices(d, self.kh, self.kw, self.co, self.hi, self.wi, p, 1)
        patched_weights= np.rot90(self.weights.transpose(3, 0, 1, 2), 2, axes=(2,3)).reshape(self.ci, -1)
        dx = (patched_weights @ patched_matrix).reshape(self.ci, self.hi, self.wi, -1).transpose(1, 2, 0, 3) # PyNN format
        dx*= self.prev_layer.dz
        return dx
        # Old code left as a reference
        # for b_ in range(b):
        #     for ci_ in range(ci):
        #         for co_ in range(co):
        #             dx[...,ci_,b_] += convolve2d(self.dx[...,co_,b_], self.weights[co_,...,ci_], mode='full')
        #         dx[...,ci_,b_] *= self.prev_layer.dz[...,ci_,b_]    

    def calculate_change(self):
        d = self.dx.transpose(3, 2, 0, 1) # b, c, h, w
        b = d.shape[0]
        if self.stride > 1: 
            mask = np.zeros((self.stride, self.stride));  mask[0,0] = 1
            d = np.kron(d, mask)[...,:-(self.stride-1),:-(self.stride-1)]
        b, c, ho, wo = d.shape
        p= self.padding
        act = np.pad(self.prev_layer.a.transpose(2, 3, 0, 1), ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
        patched_act= im2col_indices(act, ho, wo, b, self.kh, self.kw, 1)
        #patched_act= im2col_indices(act, ho, wo, b, self.kh, self.kw, self.padding, 1)
        patched_grad= d.transpose(1, 0, 2, 3).reshape(self.co, -1)
        self.changeW = (patched_grad @ patched_act).reshape(self.co, self.kh, self.kw, self.ci)    
        self.changeB = self.dx.transpose(2, 3, 0, 1).sum(axis=(1,2,3))


    def update_weights(self, eta, b):
        self.weights-= eta * self.changeW
        self.bias   -= eta * self.changeB
        # Old code left as a reference
        # for b_ in range(b):
        #     for co_ in range(co):
        #         for ci_ in range(ci):
        #             self.weights[co_,...,ci_] -= (eta/b) * \
        #                convolve2d(self.prev_layer.a[...,ci_,b_], self.dx[...,co_,b_], mode='valid')
        #         self.bias[co_] -= (eta/b) * self.dx[...,co_,b_].sum(axis=-1).sum(axis=-1)        

class Pool2D(Layer):
    """ Pool2D layer for neural network """

    def __init__(self, pool_shape=(2,2), func='max'):
        super().__init__()
        self.pool_shape = pool_shape
        self.func= getattr(np, func)

    def initialize(self,):
        h, w, c = self.prev_layer.shape
        ph, pw = block_reduce(np.zeros([h, w]), self.pool_shape, self.func).shape
        self.shape = (ph, pw, c)
        self.n = np.prod(self.shape)

    def infer(self, prev_a):
        h, w, c, b = prev_a.shape
        a = np.zeros(self.shape + (b,))
        for b_ in range(b):
            for c_ in range(c):
                a[...,c_,b_] = block_reduce(prev_a[...,c_,b_], self.pool_shape, self.func)
        return a

    def forward(self, prev_a):
        h, w, c, b = prev_a.shape
        self.a  = np.zeros(self.shape + (b,))
        self.dz = np.zeros(self.shape + (b,))
        for b_ in range(b):
            for c_ in range(c):
                self.a[...,c_,b_] = block_reduce(prev_a[...,c_,b_], self.pool_shape, self.func)
                r= np.kron(self.a[...,c_,b_], np.ones(self.pool_shape))[:prev_a[...,c_,b_].shape[0],:prev_a[...,c_,b_].shape[1]]
                mask = np.equal(prev_a[...,c_,b_], r).astype(int)
                D = mask * self.prev_layer.dz[...,c_,b_]
                self.dz[...,c_,b_] = block_reduce(D, self.pool_shape, func=np.max)
        printf("forward:", type(self).__name__, self.shape, self.a.shape)

    def get_gradient(self):
        printf("  get_gradient:", type(self).__name__, self.shape)
        b = self.a.shape[-1]
        h, w, c = self.shape
        ph, pw, c = self.prev_layer.shape
        grad = np.zeros([ph, pw, c, b])
        for b_ in range(b):
            for c_ in range(c):
                prev_a= self.prev_layer.a[...,c_,b_]
                r= np.kron(self.a[...,c_,b_], np.ones(self.pool_shape))[:prev_a.shape[0],:prev_a.shape[1]]
                mask = np.equal(prev_a, r).astype(int)
                grad[...,c_,b_] = mask * np.kron(self.dx[...,c_,b_], np.ones(self.pool_shape))[:prev_a.shape[0],:prev_a.shape[1]]
        return grad

    def calculate_change(self):
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

    def calculate_change(self):
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
        printf("  get_gradient:", type(self).__name__, self.shape)
        return self.dx * self.mask

    def calculate_change(self):
        pass

    def update_weights(self, eta, b):
        pass
