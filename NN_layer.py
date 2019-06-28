import numpy as np
import math
import random
import NN_utils
from NN_utils import printf

from scipy.signal import convolve2d
from skimage.measure import block_reduce

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

    def backward(self, grad= []):
        printf("\nbackward:", type(self).__name__, self.shape)
        if grad == []: self.d = self.next_layer.get_gradient()
        else:          self.d = grad
        printf("    ", type(self).__name__, " self.d", self.d.shape)

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
        self.D = self.act_der(z)
        printf("forward:", type(self).__name__, self.shape, self.a.shape)
        
    def get_gradient(self):
        printf("  get_gradient:", type(self).__name__, self.shape)
        return (self.weights.T @ self.d) * self.prev_layer.D

    def update_weights(self, eta, b):
        self.weights-= (eta/b) * (self.d @ self.prev_layer.a.T)
        self.bias-= (eta/b) * self.d.sum(axis=1).reshape(self.bias.shape[0], 1)
        
class Conv2D(Layer):
    """ Conv2D layer for neural network """

    def __init__(self, nfilters=1, filter_shape=(3, 3, 1), padding=0, stride=1, activation="sigmoid"):
        super().__init__()
        self.nfilters = nfilters
        self.filter_shape = filter_shape
        self.padding = padding
        self.stride = stride
        self.act= getattr(NN_utils, activation)
        self.act_der= getattr(NN_utils, "%s_derivate" % activation)          

    def initialize(self):
        self.weights = np.random.uniform(-1, 1, (self.nfilters,)+self.filter_shape)
        self.bias = np.random.uniform(-1, 1, (self.nfilters,))
        h, w, c = self.prev_layer.shape
        ph, pw = convolve2d(np.zeros([h, w]), self.weights[0,...,0], mode='valid').shape
        self.shape = (ph, pw, self.nfilters)
        self.n = np.prod(self.shape)

    def show(self):
        super().show()
        print('#Filters ', self.weights.shape)

    def infer(self, prev_a):
        #z = NN_utils.convolve_scipy(prev_a, self.weights, self.bias, self.padding, self.stride)
        z = NN_utils.convolve(prev_a, self.weights, self.bias, self.padding, self.stride) 
        return self.act(z)

    def forward(self, prev_a):
        #z = NN_utils.convolve_scipy(prev_a, self.weights, self.bias, self.padding, self.stride)
        z = NN_utils.convolve(prev_a, self.weights, self.bias, self.padding, self.stride)                
        self.a = self.act(z)
        self.D = self.act_der(z)
        printf("forward:", type(self).__name__, self.shape, self.a.shape)

    def get_gradient(self):
        printf("  get_gradient:", type(self).__name__, self.shape)
        ho, wo, co, b = self.a.shape
        hi, pi, ci = self.prev_layer.shape
        kh, kw, ci = self.filter_shape
        grad = np.zeros([hi, hi, ci, b])
        for b_ in range(b):
            for ci_ in range(ci):
                for co_ in range(co):
                    grad[...,ci_,b_] += convolve2d(self.d[...,co_,b_], self.weights[co_,...,ci_], mode='full')
                grad[...,ci_,b_] *= self.prev_layer.D[...,ci_,b_]             
        return grad

    def update_weights(self, eta, b):
        ho, wo, co = self.shape
        kh, kw, ci = self.filter_shape
        for b_ in range(b):
            for co_ in range(co):
                for ci_ in range(ci):
                    self.weights[co_,...,ci_] -= (eta/b) * \
                        np.rot90(convolve2d(self.prev_layer.a[...,ci_,b_], self.d[...,co_,b_], mode='valid'), 2)
                self.bias[co_] -= (eta/b) * self.d[...,co_,b_].sum(axis=-1).sum(axis=-1)

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
        self.a = np.zeros(self.shape + (b,))
        self.D = np.zeros(self.shape + (b,))
        for b_ in range(b):
            for c_ in range(c):
                self.a[...,c_,b_] = block_reduce(prev_a[...,c_,b_], self.pool_shape, self.func)
                r= np.kron(self.a[...,c_,b_], np.ones(self.pool_shape))[:prev_a[...,c_,b_].shape[0],:prev_a[...,c_,b_].shape[1]]
                mask = np.equal(prev_a[...,c_,b_], r).astype(int)
                D = mask * self.prev_layer.D[...,c_,b_]
                self.D[...,c_,b_] = block_reduce(D, self.pool_shape, func=np.max)
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
                grad[...,c_,b_] = mask * np.kron(self.d[...,c_,b_], np.ones(self.pool_shape))[:prev_a.shape[0],:prev_a.shape[1]]
        return grad

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
        if hasattr(self.prev_layer, 'D'):
            self.D = self.prev_layer.D.reshape(-1, b)
        else:
            self.D = np.ones(self.a.shape)
        printf("forward:", type(self).__name__, self.shape, self.a.shape)

    def get_gradient(self):
        printf("  get_gradient:", type(self).__name__, self.shape)
        b = self.a.shape[-1]
        return self.d.reshape(self.prev_layer.shape + (b,))

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
        if hasattr(self.prev_layer, 'D'):
            self.D = self.prev_layer.D * self.mask 
        else:
            self.D = np.ones(self.a.shape) * self.mask 
        printf("forward:", type(self).__name__, self.shape, self.a.shape)

    def get_gradient(self):
        printf("  get_gradient:", type(self).__name__, self.shape)
        return self.d * self.mask

    def update_weights(self, eta, b):
        pass
