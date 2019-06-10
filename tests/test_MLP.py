import random
import numpy
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from NN_model import *
from NN_layer import *

# A couple of details...
verbose_mode = True 
random.seed(0)
numpy.set_printoptions(precision=15)
numpy.random.seed(30)

# Create an instance of a MLP with 2, 2, 3 and 2 neurons in layers L1 (inputs), L2, L3 and L4 (outputs)
print('**** Creating MLP model...')

model  = Model()
model.add( Input(shape=(2)), )
model.add( FC(shape=(100), activation="sigmoid") )
model.add( FC(shape=(100), activation="sigmoid") )
model.add( FC(shape=(2), activation="sigmoid") )

model.show()

# Data to train the NN. From
#    C. Higham & D. Higham
#    "Deep Learning: An Introduction for Applied Mathematicians"
#    ArXiV: 1801.05894, 2018
x       = numpy.zeros([2, 10])
x[0,:]  = [0.1, 0.3, 0.1, 0.6, 0.4, 0.6, 0.5, 0.9, 0.4, 0.7]
x[1,:]  = [0.1, 0.4, 0.5, 0.9, 0.2, 0.3, 0.6, 0.2, 0.4, 0.6]

y        = numpy.zeros([2,10])
y[0, :]   = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
y[1, :]   = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

# Train the model
print('**** Training...')
eta     = 0.001   # Learning rate
nepochs = 1000     # Number of epochs to train
b       = 2      # Batch size
print('     Epochs:', nepochs, 'Batch size:', b, 'Learning rate:', eta)

print(model.infer(x))

model.train(x, y, eta, nepochs, b)

print(x, y)
print(model.infer(x))

print('**** Done... and thanks for all the fish!!!')

