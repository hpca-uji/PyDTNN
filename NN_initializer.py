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
import scipy.stats as stats

# Initializers

def compute_fans(shape):
    if len(shape) == 2:
        fan_in, fan_out = shape[0], shape[1]
    if len(shape) > 2:
        receptive_field = np.prod(shape[2:])
        fan_in  = shape[1] * receptive_field
        fan_out = shape[0] * receptive_field
    return fan_in, fan_out

def generate_distribution(shape, scale, mode, distribution, dtype):
    fan_in, fan_out = compute_fans(shape)
    if mode == 'fan_in':
        scale /= max(1., fan_in)
    elif mode == 'fan_out':
        scale /= max(1., fan_out)
    else:
        scale /= max(1., float(fan_in + fan_out) / 2)
    if distribution == 'normal':
        # 0.879... = scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        stddev = np.sqrt(scale) / .87962566103423978
        # Truncated normal distribution [-2*stddev, 2*stddev]
        x = stats.truncnorm(-2*stddev, 2*stddev, loc=0, scale=stddev).rvs((shape)).astype(dtype)
    else:
        limit = np.sqrt(3. * scale)
        x = np.random.uniform(-limit, limit, shape).astype(dtype)
    return x

def glorot_uniform(shape, layer):
    return generate_distribution(shape, 1.0, "fan_avg", "uniform", layer.dtype)

def glorot_normal(shape, layer):
    return generate_distribution(shape, 1.0, "fan_avg", "normal", layer.dtype)

def he_uniform(shape, layer):
    return generate_distribution(shape, 2.0, "fan_in", "uniform", layer.dtype)

def he_normal(shape, layer):
    return generate_distribution(shape, 2.0, "fan_in", "normal", layer.dtype)

def lecun_uniform(shape, layer):
    return generate_distribution(shape, 1.0, "fan_in", "uniform", layer.dtype)

def lecun_normal(shape, layer):
    return generate_distribution(shape, 1.0, "fan_in", "normal", layer.dtype)

def ones(shape, layer):
    return np.ones(shape).astype(layer.dtype)

def zeros(shape, layer):
    return np.zeros(shape).astype(layer.dtype)


