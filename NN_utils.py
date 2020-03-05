import numpy as np
import math
import random
import NN_im2col_cython
from scipy.signal import convolve2d

PYDL_EVT = 60000001
PYDL_OPS_EVT = 60000002
PYDL_NUM_EVTS = 7
PYDL_OPS_NUM_EVTS = 9
im2col_mode = "fancy"
im2col_mode = "fast"

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivate(x):
    dx = sigmoid(x)
    return dx * (1 - dx)

def relu(x):
    return (x > 0) * x

def relu_derivate(x):
    return (x >= 0) * 1.0

def tanh(x):    
    return np.tanh(x)

def tanh_derivate(x):
    return 1 - np.tanh(x) ** 2
 
def arctanh(x):
    return np.arctan(x)

def arctan_derivate(x):
    return 1 / ( 1 + x ** 2)

def log(x):
    return 1 / (1 + np.exp(-1 * x))

def log_derivate(x):
    return log(x) * ( 1 - log(x))

def softmax(x):
    ex = np.exp(x - np.max(x, axis=0).reshape(1,-1).repeat(x.shape[0], axis=0))
    return ex / ex.sum(axis=0)

def softmax_derivate(x):
    # TODO
    # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
    #s = softmax.reshape(-1,1)
    #return np.diagflat(s) - np.dot(s, s.T)
    return x

def matmul(a, b, msg=None):
    if msg: printf_trace("%s; m; %8d; n; %8d; k; %8d;" % (msg, a.shape[0], b.shape[1], a.shape[1]) )
    return a @ b

def loss(targ, pred):
    return 0.5 * np.linalg.norm(pred - targ)**2 / pred.shape[-1] # equals to b

def accuracy(targ, pred):
    targ= np.argmax(targ, axis=0)
    pred= np.argmax(pred, axis=0)
    return np.sum(np.equal(targ, pred))*100 / targ.shape[-1]

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

def im2col_fancy(x, kh, kw, c, h, w, s=1, idx=None):
    # Expected 'x' format (b, c, h, w)
    if not idx:
        idx = get_indices(x.shape, kh, kw, c, h, w, s)
    cols = x[:, idx[0], idx[1], idx[2]].transpose(1, 2, 0).reshape(kh * kw * c, -1)
    return cols, idx

def col2im_fancy(cols, x_shape, kh, kw, ho, wo, s=1, idx=None):
    b, c, h, w = x_shape    
    cols_reshaped = cols.reshape(c * kh * kw, -1, b).transpose(2, 0, 1)
    x = np.zeros((b, c, h, w), dtype=cols.dtype)
    if not idx:
        idx = get_indices(x_shape, kh, kw, c, ho, wo, s)
    np.add.at(x, (slice(None), idx[0], idx[1], idx[2]), cols_reshaped) 
    return x, idx

def im2col(x, kh, kw, c, h, w, s=1, idx=None, msg=None):
    if im2col_mode == "fast":
       cols= NN_im2col_cython.im2col_cython(x.astype(np.float64), kh, kw, h, w, 0, s)
       idx= None
    elif im2col_mode == "fancy":
       cols, idx= im2col_fancy(x, kh, kw, c, h, w, s, idx)
    if msg: printf_trace("%s; m; %8d; n; %8d; k; %8d;" % (msg, 0, cols.shape[0], cols.shape[1]) )
    return cols, None

def col2im(cols, x_shape, kh, kw, ho, wo, s=1, idx=None):
    b, c, h, w = x_shape    
    if im2col_mode == "fast":
       cols= NN_im2col_cython.col2im_cython(cols, b, c, h, w, kh, kw, 0, s)
       idx= None
    elif im2col_mode == "fancy":
       cols, idx= col2im_fancy(cols, x_shape, kh, kw, ho, wo, s, idx)
    return cols, idx

def dilate_and_pad(input, p=0, s=1):
    if s > 1 or p > 0:
        b, c, h, w = input.shape
        h_, w_ = (h+((h-1)*(s-1))+2*p, w+((w-1)*(s-1))+2*p)
        res = np.zeros([b, c, h_, w_])
        res[...,p:h_-p:s,p:w_-p:s] = input
        return res
    return input 

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

def convolve_scipy(input, weights, bias, p=0, s=1):
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

def printf_trace(*args):
    pass
    #print(*args)

def printf(*args):
    pass
    #print(*args)
