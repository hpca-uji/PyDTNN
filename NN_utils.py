import numpy as np
import math
import random

def sigmoid(x):
    """ Apply elementwise sigmoid function """
    return 1 / (1 + np.exp(-x))
    #return math.exp(-numpy.logaddexp(0, -x))

def sigmoid_derivate(x):
    """ Apply elementwise sigmoid function f(x)"""
    """ f'(x) = f(x) * (1 - f(x))"""
    x = sigmoid(x)
    return np.multiply(x, (1-x))

def relu(x):
    """ Apply elementwise ReLU function """
    x[x<0]=0
    return x

def relu_derivate(x):
    """ Apply elementwise derivative ReLU function """
    """ f'(x) = 1 if x> 0, f'(x) = 0 otherwise """
    x[x<0]=0
    x[x>0]=1
    return x

def convolution(weights, a):
    ci, kh, kw, co = weights.shape
    hi, wi, ci, b  = a.shape
    ho             = hi-kh+1
    wo             = wi-kw+1
    z = np.zeros([ho, wo, co, b])
    #### print(weights.shape)
    #### print(a.shape)
    #### print(z.shape)
    
    for i0 in range(b):
        for i1 in range(ho):
            for i2 in range(wo):
                for i3 in range(co):
                    sum = 0
                    for i4 in range(kh):
                        for i5 in range(kw):
                            for i6 in range(ci):
                                #### print(i1, i2, i3, i4, i5, i6, i1+i4, i2+i5)
                                sum += a[i1+i4][i2+i5][i6][i0] * weights[i6][i4][i5][i3]
                    z[i1][i2][i3][i0] = sum

    return z

def convolution_transpose(weights, a):
    ci, kh, kw, co = weights.shape
    hi, wi, ci, b  = a.shape
    ho             = hi-kh+1
    wo             = wi-kw+1
    z = np.zeros([ho, wo, co, b])
    #### print(weights.shape)
    #### print(a.shape)
    #### print(z.shape)
    
    for i0 in range(b):
        for i1 in range(ho):
            for i2 in range(wo):
                for i3 in range(co):
                    sum = 0
                    for i4 in range(kh):
                        for i5 in range(kw):
                            for i6 in range(ci):
                                #### print(i1, i2, i3, i4, i5, i6, i1+i4, i2+i5)
                                sum += a[i1+i4][i2+i5][i6][i0] * weights[i6][i5][i4][i3]
                    z[i1][i2][i3][i0] = sum

    return z
