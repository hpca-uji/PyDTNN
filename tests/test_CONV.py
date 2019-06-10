import random
import numpy
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from NN_model import *
from NN_layer import *

# A couple of details...
verbose_mode =  False # True
random.seed(0)
numpy.set_printoptions(precision=15)
numpy.random.seed(30)

print('**** Creating CONV model...')

in_shape= (128, 128, 3)

model = Model()
model.add( Input(shape=in_shape) )
model.add( Conv2D(nfilters=2, filter_shape=(3, 3, 3), activation="relu") )
model.add( Pool2D(pool_shape=(2,2), func='max') )
model.add( Flatten() )
model.add( FC(shape=(36,), activation="sigmoid") )
model.add( FC(shape=(1,), activation="sigmoid") )

model.show()

# Data to train the NN. 
x = numpy.random.rand(*(in_shape + (10,)))
y = numpy.random.rand(1, 10)

# Train the model
print('**** Training...')
eta     = 0.05   # Learning rate
nepochs = 10      # Number of epochs to train
b       = 3      # Batch size
print('     Epochs:', nepochs, 'Batch size:', b, 'Learning rate:', eta)

#print(model.infer(x))

model.train(x, y, eta, nepochs, b)

print('**** Done... and thanks for all the fish!!!')

