""" Python Distributed Training of Neural Networks - PyDTNN

PyDTNN is a light-weight library for distributed Deep Learning training and 
inference that offers an initial starting point for interaction with 
distributed training of (and inference with) deep neural networks. PyDTNN 
priorizes simplicity over efficiency, providing an amiable user interface 
which enables a flat accessing curve. To perform the training and inference 
processes, PyDTNN exploits distributed inter-process parallelism (via MPI) 
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


from model import *
from layer import *
from activation import *

def create_inceptionv3_cifar10(model):
    model.add( Input(shape=(3, 224, 224)) )
    model.add( Conv2D(nfilters=32, filter_shape=(3, 3), stride=2, weights_initializer="he_uniform") )
    model.add( Conv2D(nfilters=32, filter_shape=(3, 3), weights_initializer="he_uniform") )
    model.add( Conv2D(nfilters=64, filter_shape=(3, 3), padding=1, weights_initializer="he_uniform") )
    model.add( MaxPool2D(pool_shape=(3,3), stride=2) )
    model.add( Conv2D(nfilters=80,  filter_shape=(1, 1), weights_initializer="he_uniform") )
    model.add( Conv2D(nfilters=192, filter_shape=(3, 3), weights_initializer="he_uniform") )
    model.add( MaxPool2D(pool_shape=(3,3), stride=2) )

    inception_blocks = [ [64,  48, 64,  64,  96,  32],
                         [64,  48, 64,  64,  96,  64],
                         [64,  48, 64,  64,  96,  64] ]
    
    for n1x1, n5x5red, n5x5, n3x3red, n3x3, pool_planes in inception_blocks:
        model.add( 
            ConcatenationBlock( 
                [   Conv2D(nfilters=n1x1, filter_shape=(1,1), weights_initializer="he_uniform")
                ],
                [   Conv2D(nfilters=n5x5red, filter_shape=(1,1), weights_initializer="he_uniform"),
                    Conv2D(nfilters=n5x5, filter_shape=(5,5), padding=2, weights_initializer="he_uniform")
                ],
                [   Conv2D(nfilters=n3x3red, filter_shape=(1,1), weights_initializer="he_uniform"),
                    Conv2D(nfilters=n3x3, filter_shape=(3,3), padding=1, weights_initializer="he_uniform"),
                    Conv2D(nfilters=n3x3, filter_shape=(3,3), padding=1, weights_initializer="he_uniform")
                ],
                [   AveragePool2D(pool_shape=(3,3), stride=1, padding=1),
                    Conv2D(nfilters=pool_planes, filter_shape=(1,1), weights_initializer="he_uniform")
                ] ) )                      


    inception_blocks = [ [ 384,  64,  96 ] ]
    
    for n1x1, n3x3red, n3x3 in inception_blocks:
        model.add( 
            ConcatenationBlock( 
                [   Conv2D(nfilters=n1x1, filter_shape=(3,3), stride=2, padding=0, weights_initializer="he_uniform")
                ],
                [   Conv2D(nfilters=n3x3red, filter_shape=(1,1), weights_initializer="he_uniform"),
                    Conv2D(nfilters=n3x3, filter_shape=(3,3), weights_initializer="he_uniform"),
                    Conv2D(nfilters=n3x3, filter_shape=(3,3), stride=2, padding=1, weights_initializer="he_uniform")
                ],
                [   MaxPool2D(pool_shape=(3,3), stride=2, padding=0)
                ] ) )   


    inception_blocks = [ [192, 128],
                         [192, 160],
                         [192, 160],
                         [192, 192] ]
    
    for n1x1, n1x7 in inception_blocks:
        model.add( 
            ConcatenationBlock( 
                [   Conv2D(nfilters=n1x1, filter_shape=(1,1), weights_initializer="he_uniform")
                ],
                [   Conv2D(nfilters=n1x7, filter_shape=(1,1), weights_initializer="he_uniform"),
                    Conv2D(nfilters=n1x7, filter_shape=(1,7), padding=(3,0), weights_initializer="he_uniform"),
                    Conv2D(nfilters=n1x1, filter_shape=(7,1), padding=(0,3), weights_initializer="he_uniform")
                ],
                [   Conv2D(nfilters=n1x7, filter_shape=(1,1), weights_initializer="he_uniform"),
                    Conv2D(nfilters=n1x7, filter_shape=(7,1), padding=(0,3), weights_initializer="he_uniform"),
                    Conv2D(nfilters=n1x7, filter_shape=(1,7), padding=(3,0), weights_initializer="he_uniform"),
                    Conv2D(nfilters=n1x7, filter_shape=(7,1), padding=(0,3), weights_initializer="he_uniform"),
                    Conv2D(nfilters=n1x1, filter_shape=(1,7), padding=(3,0), weights_initializer="he_uniform"),
                ],
                [   AveragePool2D(pool_shape=(3,3), stride=1, padding=1),
                    Conv2D(nfilters=n1x1, filter_shape=(1,1), weights_initializer="he_uniform")
                ] ) )  


    inception_blocks = [ [192, 320] ]
    
    for n1x1, n3x3 in inception_blocks:
        model.add( 
            ConcatenationBlock( 
                [   Conv2D(nfilters=n1x1, filter_shape=(1,1), weights_initializer="he_uniform"),
                    Conv2D(nfilters=n3x3, filter_shape=(3,3), stride=2, weights_initializer="he_uniform")
                ],
                [   Conv2D(nfilters=n1x1, filter_shape=(1,1), padding=1, weights_initializer="he_uniform"),
                    Conv2D(nfilters=n1x1, filter_shape=(1,7), padding=1, weights_initializer="he_uniform"),
                    Conv2D(nfilters=n1x1, filter_shape=(7,1), padding=1, weights_initializer="he_uniform"),
                    Conv2D(nfilters=n1x1, filter_shape=(3,3), stride=2, weights_initializer="he_uniform")
                ],
                [   MaxPool2D(pool_shape=(3,3), stride=2)
                ] ) )  


    inception_blocks = [ [ 320, 384, 448, 192 ],
                         [ 320, 384, 448, 192 ] ]
    
    for n1x1b0, n1x1b1, n1x1b2, n1x1b3 in inception_blocks:
        model.add( 
            ConcatenationBlock( 
                [   Conv2D(nfilters=n1x1b0, filter_shape=(1,1), weights_initializer="he_uniform")
                ],
                [   Conv2D(nfilters=n1x1b1, filter_shape=(1,1), weights_initializer="he_uniform"),
                    ConcatenationBlock( 
                        [  Conv2D(nfilters=n1x1b1, filter_shape=(1,3), padding=(0,1), weights_initializer="he_uniform") ],
                        [  Conv2D(nfilters=n1x1b1, filter_shape=(3,1), padding=(1,0), weights_initializer="he_uniform") ] )
                ],
                [   Conv2D(nfilters=n1x1b2, filter_shape=(1,1), weights_initializer="he_uniform"),
                    Conv2D(nfilters=n1x1b1, filter_shape=(3,3), padding=1, weights_initializer="he_uniform"),
                    ConcatenationBlock( 
                        [  Conv2D(nfilters=n1x1b1, filter_shape=(1,3), padding=(0,1), weights_initializer="he_uniform") ],
                        [  Conv2D(nfilters=n1x1b1, filter_shape=(3,1), padding=(1,0), weights_initializer="he_uniform") ] )
                ],
                [   AveragePool2D(pool_shape=(3,3)),
                    Conv2D(nfilters=n1x1b3, filter_shape=(1,1), padding=1, weights_initializer="he_uniform")
                ] ) )

    model.add( AveragePool2D(pool_shape=(8,8), stride=1) ) # Global average pooling 2D
    model.add( Flatten() )
    model.add( FC(shape=(1024,)) )
    model.add( BatchNormalization() )
    model.add( Relu() )    
    model.add( FC(shape=(1000,), activation="softmax") )
    return model
