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

class Optimizer():

    def __init__(self):
        pass

    def update(self, **kwargs):
        pass


class SGD(Optimizer):

    def __init__(self, learning_rate=1e-2, momentum=0.9, nesterov=False, decay=0.0):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.decay = decay

    def update(self, layer, batch_size):
        it = getattr(layer, "it", 0)
        lr = self.learning_rate / batch_size
        if self.decay > 0: 
            lr = lr * (1. / (1. + self.decay * it))
        setattr(layer, "it", it+1)

        for w_, dw_ in zip(layer.train_vars, layer.grad_vars):
            w, dw = getattr(layer, w_), getattr(layer, dw_)
            velocity = getattr(layer, "velocity_%s" % (w_), np.zeros_like(w, dtype=layer.dtype))

            velocity = self.momentum * velocity - lr * dw
            if self.nesterov:
                w += self.momentum * velocity - lr * dw
            else:
                w += velocity

            setattr(layer, w_, w)
            setattr(layer, "velocity_%s" % (w_), velocity)


class RMSProp(Optimizer):

    def __init__(self, learning_rate=1e-2, rho=0.9, epsilon=1e-7, decay=0.0):
        super().__init__()
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon
        self.decay = decay        

    def update(self, layer, batch_size):
        it = getattr(layer, "it", 0)
        lr = self.learning_rate / batch_size
        if self.decay > 0: 
            lr = lr * (1. / (1. + self.decay * it))
        setattr(layer, "it", it+1)

        for w_, dw_ in zip(layer.train_vars, layer.grad_vars):
            w, dw = getattr(layer, w_), getattr(layer, dw_)
            cache = getattr(layer, "cache_%s" % (w_), np.zeros_like(w, dtype=layer.dtype))

            cache = self.rho * cache + (1 - self.rho) * dw**2
            w -= lr * dw / np.sqrt(cache + self.epsilon)

            setattr(layer, w_, w)
            setattr(layer, "cache_%s" % (w_), cache)


class Adam(Optimizer):

    def __init__(self, learning_rate=1e-2, beta1=0.99, beta2=0.999, epsilon=1e-7, decay=0.0):
        super().__init__()
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.decay = decay

    def update(self, layer, batch_size):
        it = getattr(layer, "it", 0)
        lr = self.learning_rate / batch_size
        if self.decay > 0: 
            lr = lr * (1. / (1. + self.decay * it))
        setattr(layer, "it", it+1)

        for w_, dw_ in zip(layer.train_vars, layer.grad_vars):
            w, dw = getattr(layer, w_), getattr(layer, dw_)
            m = getattr(layer, "m_%s" % (w_), np.zeros_like(w, dtype=layer.dtype))
            v = getattr(layer, "v_%s" % (w_), np.zeros_like(w, dtype=layer.dtype))

            m = self.beta1 * m + (1 - self.beta1) * dw
            v = self.beta2 * v + (1 - self.beta2) * dw**2

            mt = m / (1 - self.beta1**it)
            vt = v / (1 - self.beta2**it)
    
            w -= lr * mt / np.sqrt(vt + self.epsilon)

            setattr(layer, w_, w)
            setattr(layer, "m_%s" % (w_), m)
            setattr(layer, "v_%s" % (w_), v)


class Nadam(Optimizer):

    def __init__(self, learning_rate=1e-2, beta1=0.99, beta2=0.999, epsilon=1e-7, decay=0.0):
        super().__init__()
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.decay = decay

    def update(self, layer, batch_size):
        it = getattr(layer, "it", 0)
        lr = self.learning_rate / batch_size
        if self.decay > 0: 
            lr = lr * (1. / (1. + self.decay * it))
        setattr(layer, "it", it+1)

        for w_, dw_ in zip(layer.train_vars, layer.grad_vars):
            w, dw = getattr(layer, w_), getattr(layer, dw_)
            m = getattr(layer, "m_%s" % (w_), np.zeros_like(w, dtype=layer.dtype))
            v = getattr(layer, "v_%s" % (w_), np.zeros_like(w, dtype=layer.dtype))

            m = self.beta1 * m + (1 - self.beta1) * dw
            v = self.beta2 * v + (1 - self.beta2) * dw**2

            mt = (m + (1 - self.beta1) * dw) / (1 - self.beta1**it)
            vt = v / (1 - self.beta2**it)
    
            w -= lr * mt / np.sqrt(vt + self.epsilon)

            setattr(layer, w_, w)
            setattr(layer, "m_%s" % (w_), m)
            setattr(layer, "v_%s" % (w_), v)
