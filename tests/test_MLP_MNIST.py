#### test_pyNN.py

import random
import numpy
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import NN_basics

# A couple of details...
verbose_mode = False #True 
random.seed(0)
numpy.set_printoptions(precision=15)
numpy.random.seed(30)

# Create an instance of a MLP with 2, 2, 3 and 2 neurons in layers L1 (inputs), L2, L3 and L4 (outputs)
print('**** Creating MLP model...')
L0_type      = 'IN'          # Type
L0_neurons   = 784           # Number of neurons. For MNIST: 28 x 28 = 784
L0_filters   = ''            # No filters for layer L0
#
L1_type      = 'FC'         
L1_neurons   = 50        
L1_filters   = ''            
#
L2_type      = 'FC'          
L2_neurons   = 10            # For MNIST, output is a digit between 0 and 9
L2_filters   = ''          
#

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

# Data to train the NN. From MNIST
import MNIST_basics
training_dataset = MNIST_basics.read_idx('datasets/train_images_idx3_ubyte')
training_labels = MNIST_basics.read_idx('datasets/train_labels_idx1_ubyte')
train_dataset_size = 60000
xall = numpy.transpose(training_dataset.reshape(train_dataset_size, 28*28))
zall = numpy.transpose(training_labels.reshape(train_dataset_size, 1))
yall = numpy.zeros([10, train_dataset_size])
subset_size = 300
x = xall[:, 0:subset_size].copy()
y = yall[:, 0:subset_size].copy()
#### for i in range(train_dataset_size):

for k in range(subset_size):
  y[zall[0, k], k] = 1

#### x       = numpy.zeros([2, 10])
#### x[0,:]  = [0.1, 0.3, 0.1, 0.6, 0.4, 0.6, 0.5, 0.9, 0.4, 0.7]
#### x[1,:]  = [0.1, 0.4, 0.5, 0.9, 0.2, 0.3, 0.6, 0.2, 0.4, 0.6]

#### y        = numpy.zeros([2,10])
#### y[0, :]   = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
#### y[1, :]   = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

# Train the model
print('**** Training...')
eta     = 0.01   # Learning rate
nepochs = 100    # Number of epochs to train
b       = 10     # Batch size
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

print('**** Done... and thanks for all the fish!!!')

