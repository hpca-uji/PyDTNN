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
from scipy.signal import convolve2d
import scipy.linalg.blas as slb

try:
    import pycuda.autoinit
    import pycuda.gpuarray as gpuarray
    import pycuda.driver as drv
    import skcuda.linalg as culinalg
    import skcuda.misc as cumisc
except:
    pass

# Initializers

def glorot_initializer(out_shape, layer):
    lim = np.sqrt(6.0 / float((np.prod(layer.prev_layer.shape)+np.prod(layer.shape))))
    return np.random.uniform(-lim, lim, out_shape).astype(layer.dtype)
        
def zeros_initializer(out_shape, layer):
    return np.zeros(out_shape).astype(layer.dtype)


# Matmul operation

def matmul(a, b):
    #if a.dtype == np.float32: 
    #    c = slb.sgemm(1.0, a, b)
    #elif a.dtype == np.float64:
    #    c = slb.dgemm(1.0, a, b)
    #else:
    # Naive matmul gets more performance than scipy blas!
    c = a @ b
    return c

def matmul_gpu(a, b):
    if not a.flags["C_CONTIGUOUS"]: 
        a = np.ascontiguousarray(a)
    if not b.flags["C_CONTIGUOUS"]: 
        b = np.ascontiguousarray(b)
    a_gpu = gpuarray.to_gpu(a)
    b_gpu = gpuarray.to_gpu(b)
    c_gpu = culinalg.dot(a_gpu, b_gpu) 
    return c_gpu.get()

# Loss functions for classification CNNs

loss_format = {"categorical_accuracy":      prefix + "acc: %5.2f%%", 
               "categorical_cross_entropy": prefix + "cro: %.2f",
               "categorical_hinge":         prefix + "hin: %.2f",
               "categorical_mse":           prefix + "mse: %.2f",
               "categorical_mae":           prefix + "mae: %.2f"
               "regression_mse":            prefix + "mse: %.2f",
               "regression_mae":            prefix + "mae: %.2f"}

def categorical_cross_entropy(Y_pred, Y_targ):
    b = Y_targ.shape[0]
    return -np.sum(np.log(Y_pred[np.arange(b), np.argmax(Y_targ, axis=1)])) / b

def categorical_accuracy(Y_pred, Y_targ):
    b = Y_targ.shape[0]
    return np.sum(Y_targ[np.arange(b), np.argmax(Y_pred, axis=1)])*100 / b

def categorical_hinge(Y_targ, Y_pred):
    pos = K.sum(Y_targ * Y_pred, axis=-1)
    neg = K.max((1.0 - Y_targ) * Y_pred, axis=-1)
    return K.mean(K.maximum(0.0, neg - pos + 1), axis=-1)

def categorical_mse(Y_pred, Y_targ):
    b = Y_targ.shape[0]
    return np.square(1 - Y_pred[np.arange(b), np.argmax(Y_targ, axis=1)]).mean()

def categorical_mae(Y_pred, Y_targ):
    b = Y_targ.shape[0]
    targ = np.argmax(Y_targ, axis=1)
    return np.sum(np.absolute(1 - Y_pred[np.arange(b), np.argmax(Y_targ, axis=1)]))

def regression_mse(Y_pred, Y_targ):
    return np.square(Y_targ - Y_pred).mean()

def regression_mae(Y_pred, Y_targ):
    return np.sum(np.absolute(Y_targ - Y_pred))


# Some utility functions for debugging
def printf_trace(*args):
    pass
    #print(*args)

def printf(*args):
    pass
    #print(*args)

# All these functions below have been deprecated - use them with care!

# Only for fancy im2col/col2im indexing!
def get_indices(x_shape, kh, kw, c, h, w, s=1):
    #b, c, h, w = x_shape
    i0 = np.repeat(np.arange(kh), kw)
    i0 = np.tile(i0, c)
    i1 = s * np.repeat(np.arange(h), w)
    j0 = np.tile(np.arange(kw), kh * c)
    j1 = s * np.tile(np.arange(w), h)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(c), kh * kw).reshape(-1, 1)
    return (k.astype(int), i.astype(int), j.astype(int))

# Only for fancy im2col/col2im indexing!
def im2col_fancy(x, kh, kw, c, h, w, s=1, idx=None):
    # Expected 'x' format (b, c, h, w)
    if not idx:
        idx = get_indices(x.shape, kh, kw, c, h, w, s)
    cols = x[:, idx[0], idx[1], idx[2]].transpose(1, 2, 0).reshape(kh * kw * c, -1)
    return cols, idx

# Only for fancy im2col/col2im indexing!
def col2im_fancy(cols, x_shape, kh, kw, ho, wo, s=1, idx=None):
    b, c, h, w = x_shape    
    cols_reshaped = cols.reshape(c * kh * kw, -1, b).transpose(2, 0, 1)
    x = np.zeros((b, c, h, w), dtype=cols.dtype)
    if not idx:
        idx = get_indices(x_shape, kh, kw, c, ho, wo, s)
    np.add.at(x, (slice(None), idx[0], idx[1], idx[2]), cols_reshaped) 
    return x, idx

# Only for fancy im2col/col2im indexing!
def im2col_fancy(x, kh, kw, c, h, w, s=1, idx=None):
    cols, idx= im2col_fancy(x, kh, kw, c, h, w, s, idx)
    return cols, None

# Only for fancy im2col/col2im indexing!
def col2im_fancy(cols, x_shape, kh, kw, ho, wo, s=1, idx=None):
    b, c, h, w = x_shape    
    cols, idx= col2im_fancy(cols, x_shape, kh, kw, ho, wo, s, idx)
    return cols, idx

# Only for fancy im2col/col2im indexing!
def dilate_and_pad(input, p=0, s=1):
    if s > 1 or p > 0:
        b, c, h, w = input.shape
        h_, w_ = (h+((h-1)*(s-1))+2*p, w+((w-1)*(s-1))+2*p)
        res = np.zeros([b, c, h_, w_])
        res[...,p:h_-p:s,p:w_-p:s] = input
        return res
    return input 

# Only for fancy im2col/col2im indexing!
def convolve(input, weights, bias, p=0, s=1):
    h, w, ci, b    = input.shape
    co, kh, kw, ci = weights.shape
    ho = int((h + 2 * p - kh) / s + 1)
    wo = int((w + 2 * p - kw) / s + 1)
    input = input.transpose(3, 2, 0, 1) # b, c, h, w, this is needed for padding
    patched_matrix= im2col_indices(input, kh, kw, ci, ho, wo, p, s)
    patched_weights= weights.transpose(0, 3, 1, 2).reshape(co, -1)
    out = ((patched_weights @ patched_matrix).T + bias).T
    out = out.reshape(co, ho, wo, b)
    out = out.transpose(1, 2, 0, 3) # PyNN format
    return out

# Only for fancy im2col/col2im indexing!
def convolve_scipy(input, weights, bias):
    """ Does not support padding nor stride!! """
    h, w, ci, b  = input.shape  
    co, kh, kw, ci = weights.shape    
    ho = int((h + 2 * p - kh) / s + 1)
    wo = int((w + 2 * p - kw) / s + 1)      
    z = np.zeros([ho, wo, co, b])
    for b_ in range(b):
        for co_ in range(co):         
            for ci_ in range(ci):
                z[...,co_,b_] += convolve2d(input[...,ci_,b_], np.rot90(weights[co_,...,ci_], 2), mode='valid')
            z[...,co_,b_] += bias[co_]
    return z


