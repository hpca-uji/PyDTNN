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

def printf(*args):
    pass
    #print(*args)