""" Python Distributed Training of Neural Networks - PyDTNN

PyDTNN is a light-weight library for distributed Deep Learning training and 
inference that offers an initial starting point for interaction with 
distributed training of (and inference with) deep neural networks. PyDTNN 
priorizes simplicity over efficiency, providing an amiable user interface 
which enables a flat accessing curve. To perform the training and inference 
çprocesses, PyDTNN exploits distributed inter-process parallelism (via MPI) 
for clusters and intra-process (via multi-threading) parallelism to leverage 
the presence of multicore processors and GPUs at node level. For that, PyDTNN 
uses MPI4Py for message-passing, BLAS calls via NumPy for multicore processors
and PyCUDA+cuDNN+cuBLAS for NVIDIA GPUs.

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
__version__ = "1.1.0"


from NN_model import *
from NN_layer import *
from NN_activation import *

def create_googlenet_cifar10(model):
    model.add( Input(shape=(3, 32, 32)) )
    model.add( Conv2D(nfilters=192, filter_shape=(3, 3), padding=1, weights_initializer="he_uniform") )
    model.add( BatchNormalization() )
    model.add( Relu() )

    inception_blocks = [ [ 64,  96, 128, 16,  32,  32],
                         [128, 128, 192, 32,  96,  64],
                         [],
                         [192,  96, 208, 16,  48,  64],
                         [160, 112, 224, 24,  64,  64],
                         [128, 128, 256, 24,  64,  64],
                         [112, 144, 288, 32,  64,  64],
                         [256, 160, 320, 32, 128, 128],
                         [],
                         [256, 160, 320, 32, 128, 128],
                         [384, 192, 384, 48, 128, 128] ]
    
    for layout in inception_blocks:
        if layout == []:
            model.add( MaxPool2D(pool_shape=(3,3), stride=2, padding=1) )
        else:
            n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes = layout
            model.add( 
                ConcatenationBlock( 
                    [
                        # 1x1 conv branch
                        Conv2D(nfilters=n1x1, filter_shape=(1,1), weights_initializer="he_uniform"),
                        BatchNormalization(),
                        Relu()
                    ],
                    [   # 1x1 conv -> 3x3 conv branch
                        Conv2D(nfilters=n3x3red, filter_shape=(1,1), weights_initializer="he_uniform"),
                        BatchNormalization(),
                        Relu(),
                        Conv2D(nfilters=n3x3, filter_shape=(3,3), padding=1, weights_initializer="he_uniform"),
                        BatchNormalization(),
                        Relu()
                    ],
                    [   # 1x1 conv -> 5x5 conv branch
                        Conv2D(nfilters=n5x5red, filter_shape=(1,1), weights_initializer="he_uniform"),
                        BatchNormalization(),
                        Relu(),
                        Conv2D(nfilters=n5x5, filter_shape=(3,3), padding=1, weights_initializer="he_uniform"),
                        BatchNormalization(),
                        Relu(),
                        Conv2D(nfilters=n5x5, filter_shape=(3,3), padding=1, weights_initializer="he_uniform"),
                        BatchNormalization(),
                        Relu()
                    ],
                    [   # 3x3 pool -> 1x1 conv branch
                        MaxPool2D(pool_shape=(3,3), stride=1, padding=1),
                        Conv2D(nfilters=pool_planes, filter_shape=(1,1), weights_initializer="he_uniform"),
                        BatchNormalization(),
                        Relu()
                    ] ) )                      

    model.add( AveragePool2D(pool_shape=(8,8), stride=1) ) # Global average pooling 2D
    model.add( Flatten() )
    model.add( FC(shape=(1024,)) )
    model.add( BatchNormalization() )
    model.add( Relu() )    
    model.add( FC(shape=(10,), activation="softmax") )
    return model


