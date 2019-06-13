import numpy as np
import math
import random
from scipy.signal import convolve2d

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivate(x):
    dx = sigmoid(x)
    return dx * (1 - dx)

def relu(x):
    return (x > 0) * x

def relu_derivate(x):
    return (x > 0) * 1

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

def loss(targ, pred):
    return np.linalg.norm(np.linalg.norm(targ-pred))**2

def accuracy(targ, pred):
    targ= np.argmax(targ, axis=0)
    pred= np.argmax(pred, axis=0)
    return np.sum(np.equal(targ, pred))*100/targ.shape[0]

def get_im2col_indices(x_shape, kh, kw, ho, wo, p=1, s=1):
    b, c, h, w = x_shape
    i0 = np.repeat(np.arange(kh), kw)
    i0 = np.tile(i0, c)
    i1 = s * np.repeat(np.arange(ho), wo)
    j0 = np.tile(np.arange(kw), kh * c)
    j1 = s * np.tile(np.arange(wo), ho)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(c), kh * kw).reshape(-1, 1)
    return (k.astype(int), i.astype(int), j.astype(int))

def im2col_indices(x, kh, kw, ho, wo, p=1, s=1):
    x = x.transpose(3, 2, 0, 1) # b, c, h, w, this is needed for padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
    k, i, j = get_im2col_indices(x.shape, kh, kw, ho, wo, p, s)
    cols = x_padded[:, k, i, j]    
    ci = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(kh * kw * ci, -1)
    return cols

def convolve(input, weights, bias, p=0, s=1):
    h, w, ci, b    = input.shape
    co, kh, kw, ci = weights.shape
    ho = int((h + 2 * p - kh) / s + 1)
    wo = int((w + 2 * p - kw) / s + 1)
    patched_matrix= im2col_indices(input, kh, kw, ho, wo, p, s)
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

def printf(*args):
    pass
    #print(*args)
