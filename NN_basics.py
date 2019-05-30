import numpy as np
import math
import random
import NN_utils

class NN_Model:
    """ Neural network (NN).  attributes: L->number of layers, layers """

    def __init__(self, layers = ['IN', 'FC', 'FC', 'FC'], 
                       neurons = [4, 3, 5, 4, 2], 
                       filters = [None, None, None, None]):

        self.L = len(neurons)     # Number of layers
        self.nneurons  = neurons  # Neurons per layer
        self.layer_t   = layers   # Layer types
        self.dfilters  = [None]   # Dimension of filters
        self.ninputs   = [None]   # Inputs for each layer (can be derived from self.neurons)
        self.noutputs  = [None]   # Outputs for each layer (can be derived from self.neurons)
        self.weights   = [None]   # Weights/filters 
        self.bias      = [None]   # Biases 

        no = self.get_noutputs_at_layer(0)
        self.noutputs.append(no)                            

        for l in range(1, self.L):
            ni = self.get_ninputs_at_layer(l)
            self.ninputs.append(ni)                            # Number of inputs/activations  for layer l = neurons at layer l-1

            no = self.get_noutputs_at_layer(l)
            self.noutputs.append(no)                           # Number of outputs/activations for layer l = neurons at layer l

            if self.layer_t[l] == 'FC':
                self.weights.append((np.random.rand(no, ni)-0.5)*2.0) # Initialize weights for layer l, in [-1,1]
                self.bias.append((np.random.rand(no, 1)-0.5)*2.0)     # Initialize bias for layer l, in [-1,1]
                self.dfilters.append(None)
                #### print('W', l, model.weights[l].shape)
                #### print('b', l, model.bias[l].shape)

            elif self.layer_t[l] == 'CONV':
                self.dfilters.append(filters[l])
                self.weights.append(np.random.rand(ni[2], filters[l][1], filters[l][2], no[2])) # Initialize weights for layer l
                ### EQO: Do not include bias for the moment
                ### model.bias.append(np.random.rand(no, 1))     # Initialize bias for layer l

        print('----------------------------')
        l = 0
        print('Layer    ', 0)
        print('Type     ', self.layer_t[l])
        print('#Inputs  ', self.ninputs[l])
        print('#Neurons ', self.nneurons[l])
        print('#Outputs ', self.noutputs[l])
        print('#Filters ', self.dfilters[l])
        print('----------------------------')

        for l in range(1, self.L):
            print('Layer    ', l)
            print('Type     ', self.layer_t[l])
            print('#Inputs  ', self.ninputs[l])
            print('#Neurons ', self.nneurons[l])
            print('#Outputs ', self.noutputs[l])
            if (self.layer_t[l] == 'CONV'):
                print('#Filters ', self.weights[l].shape)
            print('----------------------------')

    def infer(self, sample):
        """ Inference """
        z = sample
        for l in range(1, self.L):
            if (self.layer_t[l] == 'FC'):
                a = z # Input activations for this iteration
                a = np.matmul(self.weights[l], a) + self.bias[l]
                z = NN_utils.sigmoid(a) # Non-linear function 
                #print('a',l,z)

            elif (self.layer_t[l] == 'CONV'):
                a = z
                a = NN_utils.convolution(self.weights[l], a)
                z = NN_utils.sigmoid(a) # Non-linear function 
        return z

    def train(self, samples, labels, eta, nepochs, b):
        """ SGD over all samples, in batches of size b """

        nsamples = samples.shape[1] # Numer of samples
        savecost = []               # Error after each epoch training
        batched_training_order = [] 

        for counter in range(nepochs):
            print('------- Epoch',counter+1)

            s = list(range(nsamples))
            random.shuffle(s)         # shuffle for random ordering of the batch

            counter3 = 1
            for counter2 in range(0, nsamples, b):
                nb = min(nsamples-counter2+1, b)       # Number of samples in current batch
                indices = s[counter2:counter2+nb]      # Indices into samples for current batch
                indices1 = list(map(lambda x:x+1, indices))   # Add one to use indices starting at 1
                batched_training_order += indices1     # Save access order to samples. 

                #### print('Indices', indices)
                if self.get_layer_t_at_layer(1) == 'CONV':
                    batch_samples = samples[:,:,:,indices].copy()    # Current batch samples
                else:
                    batch_samples = samples[:,indices].copy()        # Current batch samples
                if self.get_layer_t_at_layer(self.L-1) == 'FC':
                    batch_labels  = labels[:,indices].copy()         # Current batch labels
                else:
                    batch_labels  = labels[:,:,:,indices].copy()     # Current batch labels
                #### print('Samples', batch_samples.shape)
                #### print('Labels ', batch_labels.shape)
     
                # Train the model w.r.t. current batch
                self.train_batch(batch_samples, batch_labels, eta)
           
                #### for l in range(1,self.L):
                    #### print('   Layer', l)
                    #### print('Weights', trained_model.weights[l])
                    #### print('Bias',    trained_model.bias[l])

                savecost.append(self.evaluate_cost(samples, labels))
                #### savecost += [NN_Evaluate_cost(self, samples, labels)]
                print('            Batch', counter3, 'samples', indices1, 'Cost fnct:', savecost[-1])
                counter3 = counter3+1

        print('**** Access order to samples during training')
        print(batched_training_order)


    def train_batch(self, batch_samples, batch_labels, eta):
        """ Single step (batched) SGD """

        b = batch_samples.shape[1]  # Batch size = number of columns in the batch
        a = []                      # Layer input and output (after nonlinear function)
        d = []                      # Layer delta (for Gradient update)
        z = [ [] ]                  # Layer outputs, before nonlinear function
        D = [ [] ]                  # For the derivates (nonzero entries of diagonal matrix)

        a.append(batch_samples)
        d.append(np.zeros(batch_samples.shape))   # Place-holder
        # Forward pass
        #### print('a', 1, a[1].shape)
        for l in range(1, self.L):
            #### print('#### Layer', l)

            if self.layer_t[l] == 'FC':
                #### print('FP.Layer', l)
                #### print('W', l, self.weights[l].shape)
                #### print('b', l, self.bias[l].shape)
                z.append(np.matmul(self.weights[l], a[l-1]) + self.bias[l])
                a.append(NN_utils.sigmoid(z[l]))             # Non-linear function 
                #### print('a',l,a[l].shape)
                d.append(np.zeros(a[l].shape))   # Place-holder
                D.append(NN_utils.sigmoid_derivate(z[l]))

            elif self.layer_t[l] == 'CONV':
                z.append(NN_utils.convolution(self.weights[l], a[l-1]))
                a.append(NN_utils.sigmoid(z[l]))             # Non-linear function 
                #### print('a',l,a[l].shape)
                d.append(np.zeros(a[l].shape))   # Place-holder
                D.append(NN_utils.sigmoid_derivate(z[l]))

        # Back propagation. Gradient computation (GC)
        if self.layer_t[l] == 'FC':
            d[self.L-1] = np.multiply(D[self.L-1], (a[self.L-1] - batch_labels))

        elif self.layer_t[l] == 'CONV':
            #### EQO: Check this!!! Back-propagation of last layer for convolutional layer
            d[self.L-1] = np.multiply(D[self.L-1], (a[self.L-1] - batch_labels))

        for l in range(self.L-2, 0, -1):
            #### print('BP-GC.Layer', l)
            if self.layer_t[l] == 'FC':
                d[l] = np.multiply(D[l], np.matmul(np.transpose(self.weights[l+1]), d[l+1]))

            elif self.layer_t[l] == 'CONV':
                d[l] = NN_utils.convolution_transpose(self.weights[l+1], d[l+1])
                #### print('GC d', l, d[l].shape)

        # Back propagation. Weight update (WU)
        #### EQO: The weight and bias updates need to be double-checked. Not sure they should be 
        ####      divided by b (batch size)
        for l in range(self.L-1, 0, -1):
            #### print('BP-WU.Layer', l)
            if (self.layer_t[l] == 'FC'):
                self.bias[l].shape
                self.bias[l] = self.bias[l] - (eta/b) * d[l].sum(axis=1).reshape(self.bias[l].shape[0],1)
                #### print('After WU b', l, model.bias[l].shape)
                self.weights[l] = self.weights[l] - (eta/b) * np.matmul(d[l], np.transpose(a[l-1]))
                #### print('        W', l, model.weights[l])
                #### print('        b', l, model.bias[l])


    def evaluate_cost(self, samples, labels):
        """ Evaluate cost function, use inference routine"""

        nsamples = samples.shape[1] # Numer of samples
        costvec  = np.zeros([nsamples, 1])
        if self.get_layer_t_at_layer(1) == 'CONV':
            n1     = samples.shape[0]*samples.shape[1]*samples.shape[2]  # Neurons in first layer
        else:
            n1     = samples.shape[0]                                    # Neurons in first layer

        if self.get_layer_t_at_layer(self.L-1) == 'CONV':
            nL     = labels.shape[0]*labels.shape[1]*labels.shape[2]     # Neurons in last layer
        else:
            nL     = labels.shape[0]                                     # Neurons in first layer

        for i in range(nsamples):
            if self.get_layer_t_at_layer(1) == 'CONV':
                sample = samples[:,:,:,i].copy()
                sample     = sample.reshape(samples.shape[0], samples.shape[1], samples.shape[2], 1)
            else:
                sample = samples[:,i].copy()         
                sample     = sample.reshape(n1, 1)

            #### print('sample', sample)
            if self.get_layer_t_at_layer(self.L-1) == 'FC':
                label  = labels[:,i].copy()         
            else:
                label  = labels[:,:,:,i].copy()

            label      = label.reshape(nL, 1)
            #### print('label', label)
            z          = self.infer(sample)
            z.reshape(nL, 1)
            #### print('z', z)
            costvec[i] = np.linalg.norm(label - z)
            #### print('costvec', costvec[i])

        costval = np.linalg.norm(costvec)
        return (costval*costval)

        #### print(Niter, savecost[counter])


    def get_ninputs_at_layer(self, l):
        """ Number of input activations for layer l = neurons at layer l-1 """
        return self.nneurons[l-1]

    def get_noutputs_at_layer(self, l):
        """ Number of output activations for layer l = neurons at layer l """
        return self.nneurons[l]

    def get_bias_at_layer(self, l):
        """ Biases for layer l """
        return self.bias[l]

    def get_weights_at_layer(self, l):
        return self.weights[l]
        
    def get_layer_t_at_layer(self, l):
        return self.layer_t[l]

