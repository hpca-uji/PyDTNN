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

def create_resnet152_cifar10(model):
    model.add( Input(shape=(3, 32, 32)) )
    model.add( Conv2D(nfilters=64, filter_shape=(3, 3), stride=1, padding=1, weights_initializer="he_uniform") )
    model.add( BatchNormalization() )

    expansion = 4
    layout = [ [64, 3, 1], [128, 8, 2], [256, 36, 2], [512, 3, 2] ] # Resnet-152
    for n_filt, res_blocks, stride in layout:
    	for r in range(res_blocks):
            if r > 0: stride = 1
            model.add( 
                AdditionBlock( 
                    [
                        Conv2D(nfilters=n_filt, filter_shape=(1, 1), stride=1, weights_initializer="he_uniform"),
                        BatchNormalization(),
                        Relu(),
                        Conv2D(nfilters=n_filt, filter_shape=(3, 3), stride=stride, padding=1, weights_initializer="he_uniform"),
                        BatchNormalization(),
                        Relu(),
                        Conv2D(nfilters=n_filt * expansion, filter_shape=(1, 1), stride=1, weights_initializer="he_uniform"),
                        BatchNormalization() 
                    ],
                    [
                        Conv2D(nfilters=n_filt * expansion, filter_shape=(1, 1), stride=stride, weights_initializer="he_uniform"),
                        BatchNormalization() 
                    ] if r==0 or stride != 1 else [] ) )
            model.add( Relu() )    

    model.add( AveragePool2D(pool_shape=(0,0)) ) # Global average pooling 2D
    model.add( Flatten() )
    model.add( FC(shape=(512 * expansion,) ) )
    model.add( BatchNormalization() )
    model.add( Relu() )    
    model.add( FC(shape=(10,), activation="softmax") )
    return model
