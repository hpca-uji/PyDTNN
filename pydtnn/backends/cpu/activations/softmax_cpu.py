#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021-25 Universitat Jaume I
#
#  PyDTNN is free software: you can redistribute it and/or modify it under the
#  terms of the GNU General Public License as published by the Free Software
#  Foundation, either version 3 of the License, or (at your option) any later
#  version.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT
#  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
#  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
#  License for more details.
#
#  You should have received a copy of the GNU General Public License along
#  with this program. If not, see <https://www.gnu.org/licenses/>.
#

import numpy as np

from pydtnn.activations.softmax import Softmax
from pydtnn.backends.cpu.activations.activation_cpu import ActivationCPU

class SoftmaxCPU(ActivationCPU, Softmax):

    def __init__(self, shape:np.ndarray = (1,)):
        super().__init__(shape)
    
    def initialize(self, prev_shape):
        super().initialize(prev_shape)
        self.y:np.ndarray
        self._y = np.empty(shape=(self.model.batch_size, *self.shape), 
                           dtype=self.model.dtype)
        self.mul_dy = np.empty(shape=(self.model.batch_size, *self.shape),
                               dtype=self.model.dtype)

        self.axis_dim = 1
        shape_intermediate_ops = list(self.shape)
        shape_intermediate_ops[self.axis_dim-1] = 1

        self.max_x = np.empty(shape=(self.model.batch_size, *shape_intermediate_ops),
                              dtype=self.model.dtype)
        self.sum_y = np.empty(shape=(self.model.batch_size, *shape_intermediate_ops),
                              dtype=self.model.dtype)        
        self.sum_dy = np.empty(shape=(self.model.batch_size, *shape_intermediate_ops),
                               dtype=self.model.dtype)
        

    def forward(self, x:np.ndarray) -> np.ndarray:
        #self.y = np.exp(x - np.max(x, axis=1, keepdims=True))
        #self.y /= np.sum(self.y, axis=1, keepdims=True)
        #return self.y
        self.y = self._y[:x.shape[0], :]
        max_x = self.max_x[:x.shape[0], :]
        sum_y = self.sum_y[:x.shape[0], :]

        np.max(x, axis=self.axis_dim, keepdims=True, out=max_x)
        np.subtract(x, max_x, out=x)
        np.exp(x, out=self.y)
        np.sum(self.y, axis=self.axis_dim, keepdims=True, out=sum_y)
        np.divide(self.y, sum_y, out=self.y)
        
        self.y = self.y.astype(dtype=self.model.dtype)
        return self.y

    def backward(self, dy:np.ndarray) -> np.ndarray:        
        #return self.y * (dy - (dy * self.y).sum(axis=1, keepdims=True))
        sum_dy = self.sum_dy[:dy.shape[0], :]
        mul_dy = self.mul_dy[:dy.shape[0], :]

        np.multiply(dy, self.y, out=mul_dy)
        mul_dy.sum(axis=self.axis_dim, keepdims=True, out=sum_dy)
        np.subtract(dy, sum_dy, out=dy)
        np.multiply(self.y, dy, out=self.y)
        
        return self.y
