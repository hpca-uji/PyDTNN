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
from NN_layer import Layer

def SGD(layer, params):
    learning_rate = getattr(params, "learning_rate", 1e-2) 

    layer.weights -= learning_rate * layer.dw
    layer.bias    -= learning_rate * layer.db

def SGDMomentum(layer, params):
    learning_rate = getattr(params, "learning_rate", 1e-2)  
    momentum      = getattr(params, "momentum", 0.9)

    layer.velocity_w = getattr(layer, "velocity_w", np.zeros_like(layer.weights).astype(layer.dtype))
    layer.velocity_b = getattr(layer, "velocity_b", np.zeros_like(layer.bias).astype(layer.dtype))

    layer.velocity_w = momentum * layer.velocity_w - learning_rate * layer.dw
    layer.velocity_b = momentum * layer.velocity_b - learning_rate * layer.db

    layer.weights += layer.velocity_w
    layer.bias    += layer.velocity_b

def RMSProp(layer, params):
    learning_rate = getattr(params, "learning_rate", 1e-2)  
    decay_rate    = getattr(params, "decay_rate", 0.99)
    epsilon       = getattr(params, "epsilon", 1e-8)

    layer.cache_w = getattr(layer, "cache_w", np.zeros_like(layer.weights).astype(layer.dtype))
    layer.cache_b = getattr(layer, "cache_b", np.zeros_like(layer.bias).astype(layer.dtype))

    layer.cache_w = decay_rate * layer.cache_w + (1 - decay_rate) * layer.dw**2
    layer.cache_b = decay_rate * layer.cache_b + (1 - decay_rate) * layer.db**2

    layer.weights -= learning_rate * layer.dw / np.sqrt(layer.cache_w + epsilon)
    layer.bias    -= learning_rate * layer.db / np.sqrt(layer.cache_b + epsilon)

def Adam(layer, params):
    learning_rate = getattr(params, "learning_rate", 1e-2)  
    beta1         = getattr(params, "beta1", 0.99)
    beta2         = getattr(params, "beta2", 0.999)
    epsilon       = getattr(params, "epsilon", 1e-8)

    layer.m_w  = getattr(layer, "m_w", np.zeros_like(layer.weights).astype(layer.dtype))
    layer.v_w  = getattr(layer, "v_w", np.zeros_like(layer.weights).astype(layer.dtype))
    layer.m_b  = getattr(layer, "m_b", np.zeros_like(layer.bias).astype(layer.dtype))
    layer.v_b  = getattr(layer, "v_b", np.zeros_like(layer.bias).astype(layer.dtype))
    layer.iter = getattr(layer, "iter", 0)

    layer.iter+= 1
    layer.m_w = beta1 * layer.m_w + (1 - beta1) * layer.dw
    layer.v_w = beta2 * layer.v_w + (1 - beta2) * layer.dw**2
    layer.m_b = beta1 * layer.m_b + (1 - beta1) * layer.db
    layer.v_b = beta2 * layer.v_b + (1 - beta2) * layer.db**2

    mt_w = layer.m_w / (1 - beta1**layer.iter)
    vt_w = layer.v_w / (1 - beta2**layer.iter)
    mt_b = layer.m_b / (1 - beta1**layer.iter)
    vt_b = layer.v_b / (1 - beta2**layer.iter)

    layer.weights -= learning_rate * mt_w / (np.sqrt(vt_w + epsilon))
    layer.bias    -= learning_rate * mt_b / (np.sqrt(vt_b + epsilon))

