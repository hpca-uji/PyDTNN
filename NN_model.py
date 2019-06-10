import numpy as np
import math
import random
import NN_utils

from scipy.signal import convolve2d

class Model:
    """ Neural network (NN) """

    def __init__(self):
        self.layers = []
        
    def show(self):
        print('----------------------------')
        for l in self.layers:
            l.show()
            print('----------------------------')

    def add(self, layer):
        if len(self.layers) > 0:          
            self.layers[-1].next_layer = layer
            layer.initialize(self.layers[-1])
        self.layers.append(layer)

    def infer(self, sample):
        """ Inference """
        z = sample
        for l in self.layers[1:]:
            z = l.infer(z)
        return z

    def train_batch(self, batch_samples, batch_labels, eta):
        """ Single step (batched) SGD """

        b = batch_samples.shape[-1]  # Batch size = number of columns in the batch

        # Forward pass (FP)
        self.layers[0].a = batch_samples
        for l in range(1, len(self.layers)):
            self.layers[l].forward(self.layers[l-1].a)

        # Back propagation. Gradient computation (GC)
        self.layers[-1].backward((self.layers[-1].a - batch_labels))
        for l in range(len(self.layers)-2, 0, -1):
            self.layers[l].backward()
            
        # Weight update (WU)
        for l in range(len(self.layers)-1, 0, -1):
            self.layers[l].update_weights(eta, b)

    def train(self, samples, labels, eta, nepochs, b, loss_func= "loss", early_stop= True):
        """ SGD over all samples, in batches of size b """

        nsamples = samples.shape[-1] # Numer of samples
        savecost = []                # Error after each epoch training
        loss_func_= getattr(NN_utils, loss_func)
       
        for counter in range(nepochs):
            print('------- Epoch',counter+1)
            s = list(range(nsamples))
            random.shuffle(s)         # shuffle for random ordering of the batch

            counter3 = 1    
            for counter2 in range(0, nsamples, b):
                nb = min(nsamples-counter2+1, b)       # Number of samples in current batch
                indices = s[counter2:counter2+nb]      # Indices into samples for current batch

                batch_samples = samples[...,indices]    # Current batch samples
                batch_labels  = labels[...,indices]     # Current batch labels

                self.train_batch(batch_samples, batch_labels, eta)

                savecost.append(loss_func_(labels, self.infer(samples)))

                print('            Batch', counter3, "Cost fnct (%s): " % loss_func, savecost[-1])
                counter3 = counter3+1

        print('**** Access order to samples during training')


