import random
import numpy
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from NN_model import *
from NN_layer import *

# A couple of details...
random.seed(0)
numpy.set_printoptions(precision=15)
numpy.random.seed(30)

print('**** Creating CONV model...')

model = Model()
# model.add( Input(shape=(28, 28, 1)) )
# model.add( Conv2D(nfilters=64, filter_shape=(3, 3, 1), activation="relu") )
# model.add( Pool2D(pool_shape=(2,2), func='max') )
# model.add( Flatten() )
# model.add( FC(shape=(10,), activation="softmax") )

model.add( Input(shape=(28, 28, 1)) )
model.add( Conv2D(nfilters=2, filter_shape=(3, 3, 1), activation="relu") )
model.add( Pool2D(pool_shape=(2,2), func='max') )
model.add( Conv2D(nfilters=4, filter_shape=(3, 3, 2), activation="sigmoid") )
model.add( Pool2D(pool_shape=(2,2), func='max') )
model.add( Flatten() )
#model.add( FC(shape=(128,), activation="sigmoid") )
#model.add( Dropout(prob=0.5) )
#model.add( FC(shape=(36,), activation="sigmoid") )
model.add( FC(shape=(10,), activation="sigmoid") )

model.show()

# Data to train the NN. From MNIST
import MNIST_basics
training_dataset = MNIST_basics.read_idx('datasets/train_images_idx3_ubyte')
training_labels = MNIST_basics.read_idx('datasets/train_labels_idx1_ubyte')
train_dataset_size = 60000
xall = numpy.transpose(training_dataset.reshape(train_dataset_size, 1, 28, 28))
zall = numpy.transpose(training_labels.reshape(train_dataset_size, 1))
yall = numpy.zeros([10, train_dataset_size])

for k in range(train_dataset_size):
  yall[zall[0, k], k] = 1

subset_size = 300
x = xall[...,:subset_size].copy()
y = yall[...,:subset_size].copy()

targ= np.argmax(y, axis=0)
pred= np.argmax(model.infer(x), axis=0)
print("Accuracy: %.2f %%" % (np.sum(np.equal(targ, pred))*100/targ.shape[0]))
print(np.sum(np.equal(targ, pred)), targ.shape[0])

# Train the model
print('**** Training...')
eta     = 0.1   # Learning rate
nepochs = 100     # Number of epochs to train
b       = 64      # Batch size
print('     Epochs:', nepochs, 'Batch size:', b, 'Learning rate:', eta)

#print(model.infer(x))
model.train(x, y, eta, nepochs, b, loss_func="accuracy")

targ= np.argmax(y, axis=0)
pred= np.argmax(model.infer(x), axis=0)
print("Accuracy: %.2f %%" % (np.sum(np.equal(targ, pred))*100/targ.shape[0]))
print(np.sum(np.equal(targ, pred)), targ.shape[0])


print('**** Done... and thanks for all the fish!!!')

