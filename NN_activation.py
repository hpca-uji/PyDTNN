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
__version__ = "1.0.1"


import numpy as np
from NN_layer import Layer
from NN_relu_cython import relu_cython
import time


class Sigmoid(Layer):

    def __init__(self, shape=(1,)):
        super().__init__(shape)

    def forward(self, prev_a, comm=None):
        self.a  = 1 / (1 + np.exp(-prev_a))
        self.dx = (self.a * (1 - self.a))

    def backward(self, prev_dx):
        return (self.a * (1 - self.a)) * prev_dx

class Relu(Layer):

    def __init__(self, shape=(1,)):
        super().__init__(shape)

    def forward(self, prev_a, comm=None):
        self.a, self.mask = relu_cython(prev_a)

    def backward(self, prev_dx):
        return prev_dx * self.mask

class Tanh(Layer):

    def __init__(self, shape=(1,)):
        super().__init__(shape)

    def forward(self, prev_a, comm=None):
        self.a = np.tanh(prev_a)

    def backward(self, prev_dx):
        return 1 - np.tanh(prev_dx) ** 2

class Arctanh(Layer):

    def __init__(self, shape=(1,)):
        super().__init__(shape)

    def forward(self, prev_a, comm=None):
        return np.arctan(prev_a)

    def backward(self, prev_dx):
        return 1 / ( 1 + prev_dx ** 2)

class Log(Layer):

    def __init__(self, shape=(1,)):
        super().__init__(shape)

    def forward(self, prev_a, comm=None):
        return 1 / (1 + np.exp(-1 * prev_a))

    def backward(self, prev_dx):
        return log(prev_dx) * ( 1 - log(prev_dx))
    
class Softmax(Layer):

    def __init__(self, shape=(1,)):
        super().__init__(shape)

    def forward(self, prev_a, comm=None):
        self.a = np.exp(prev_a - np.max(prev_a, axis=1, keepdims=True))
        self.a /= np.sum(self.a, axis=1, keepdims=True)
       
    def backward(self, prev_dx):
        return prev_dx

# Compatibility aliases

sigmoid = Sigmoid
relu = Relu
tanh = Tanh
arctanh = Arctanh
log = Log
softmax = Softmax
