import random
import numpy
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from NN_model import *
from NN_layer import *

def show_number(num):
    ASCII_CHARS = ['.',',',':',';','+','*','?','%','S','#','@']
    for i in range(28):
        for j in range(28):
            print(ASCII_CHARS[int((float(num[i*28+j])/255.)*(len(ASCII_CHARS)-1))], end=" ")
        print()
    
# A couple of details...
verbose_mode = False #True 
random.seed(0)
numpy.set_printoptions(precision=15)
numpy.random.seed(30)

# Create an instance of a MLP with 2, 2, 3 and 2 neurons in layers L1 (inputs), L2, L3 and L4 (outputs)
print('**** Creating MLP model...')

model = Model()
model.add( Input(shape=(784)), )
model.add( FC(shape=(300), activation="sigmoid") )
model.add( FC(shape=(200), activation="sigmoid") )
model.add( FC(shape=(10), activation="softmax") )

# Data to train the NN. From MNIST
import MNIST_basics
training_dataset = MNIST_basics.read_idx('datasets/train_images_idx3_ubyte')
training_labels = MNIST_basics.read_idx('datasets/train_labels_idx1_ubyte')
train_dataset_size = 60000
xall = numpy.transpose(training_dataset.reshape(train_dataset_size, 28*28))
zall = numpy.transpose(training_labels.reshape(train_dataset_size, 1))
yall = numpy.zeros([10, train_dataset_size])

for k in range(train_dataset_size):
  yall[zall[0, k], k] = 1

subset_size = 300
x = xall[:, 0:subset_size].copy()
y = yall[:, 0:subset_size].copy()

targ= np.argmax(y, axis=0)
pred= np.argmax(model.infer(x), axis=0)
print("Accuracy: %.2f %%" % (np.sum(np.equal(targ, pred))*100/targ.shape[0]))
print(np.sum(np.equal(targ, pred)), targ.shape[0])

# Train the model
print('**** Training...')
eta     = 0.1   # Learning rate
nepochs = 200    # Number of epochs to train
b       = 64    # Batch size
print('     Epochs:', nepochs, 'Batch size:', b, 'Learning rate:', eta)
model.train(x, y, eta, nepochs, b, loss_func="accuracy")

pred= np.argmax(model.infer(x), axis=0)
print("Accuracy: %.2f %%" % (np.sum(np.equal(targ, pred))*100/targ.shape[0]))
print(np.sum(np.equal(targ, pred)), targ.shape[0])


print('**** Done... and thanks for all the fish!!!')

