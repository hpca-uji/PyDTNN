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


from NN_model import *
from NN_layer import *
from NN_activation import *

def create_vgg3do2(model):
    model.add( Input(shape=(3, 32, 32)) )
    model.add( Conv2D(nfilters=32, filter_shape=(3, 3), padding=1, activation="relu", weights_initializer="he_uniform") )
    model.add( Conv2D(nfilters=32, filter_shape=(3, 3), padding=1, activation="relu", weights_initializer="he_uniform") )
    model.add( MaxPool2D(pool_shape=(2,2), stride=2) )
    model.add( Dropout(rate=0.2) )
    model.add( Conv2D(nfilters=64, filter_shape=(3, 3), padding=1, activation="relu", weights_initializer="he_uniform") )
    model.add( Conv2D(nfilters=64, filter_shape=(3, 3), padding=1, activation="relu", weights_initializer="he_uniform") )
    model.add( MaxPool2D(pool_shape=(2,2), stride=2) )
    model.add( Dropout(rate=0.3) )
    model.add( Conv2D(nfilters=128,filter_shape=(3, 3), padding=1, activation="relu", weights_initializer="he_uniform") )
    model.add( Conv2D(nfilters=128,filter_shape=(3, 3), padding=1, activation="relu", weights_initializer="he_uniform") )
    model.add( MaxPool2D(pool_shape=(2,2), stride=2) )
    model.add( Dropout(rate=0.4) )
    model.add( Flatten() )
    model.add( FC(shape=(128,), activation="relu", weights_initializer="he_uniform") )
    model.add( Dropout(rate=0.5) )
    model.add( FC(shape=(10,), activation="softmax") )
    return model
