#### test_pyNN.py

import random
import numpy
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import NN_basics

# A couple of details...
verbose_mode =  False # True
random.seed(0)
numpy.set_printoptions(precision=15)
numpy.random.seed(30)

# Create an instance of a MLP with 2, 2, 3 and 2 neurons in layers L1 (inputs), L2, L3 and L4 (outputs)
print('**** Creating CONV model...')
L0_type      = 'IN'          # Type
L0_neurons   = [6, 6, 5]     # Number of neurons
L0_filters   = ''            # No filters for layer L0
#
L1_type      = 'CONV'         
L1_neurons   = [4, 4, 2] 
L1_filters   = [5, 3, 3, 2]  # Dimensions of filter 
#
L2_type      = 'CONV'
L2_neurons   = [3, 3, 1] 
L2_filters   = [2, 2, 2, 1]

model  = NN_basics.NN_Model([L0_type, L1_type, L2_type],
                                   [L0_neurons, L1_neurons, L2_neurons], 
                                   [L0_filters, L1_filters, L2_filters])
if verbose_mode:
    for l in range(1,model.L):
        print('%Layer', l)
        print('%-----------------------------------------------------------------')
        print('%Weights', l)
        print(model.weights[l])
        print('%Bias', l)
        print(model.bias[l])
    print('%-----------------------------------------------------------------')

# Data to train the NN. 
x       = numpy.random.rand(6, 6, 5, 10)

y        = numpy.random.rand(3,3, 1, 10)

# Train the model
print('**** Training...')
eta     = 0.05   # Learning rate
nepochs = 1      # Number of epochs to train
b       = 1      # Batch size
print('     Epochs:', nepochs, 'Batch size:', b, 'Learning rate:', eta)

model.train(x, y, eta, nepochs, b)

if verbose_mode:
    for l in range(1,model.L):
        print('%Layer', l)
        print('%-----------------------------------------------------------------')
        print('%Weights', l)
        print(model.weights[l])
        print('%Bias', l)
        print(model.bias[l])
    print('%-----------------------------------------------------------------')

# Inference with the model
#### print('**** Inference...')
#### sample = x[:,:,:,0:b].copy()
#### z = NN_basics.NN_Infer(model, sample)
#### print(z)

print('**** Done... and thanks for all the fish!!!')

