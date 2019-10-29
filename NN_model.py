import numpy as np
import math
import random
import NN_utils

from mpi4py import MPI

from scipy.signal import convolve2d

class Model:
    """ Neural network (NN) """

    def __init__(self, comm):
        self.layers = []
        self.comm = comm
        if self.comm == None:
            self.rank = 0
            self.nprocs = 1
        else:
            self.rank = self.comm.Get_rank()
            self.nprocs = self.comm.Get_size()
        
    def show(self):
        print('----------------------------')
        for l in self.layers:
            l.show()
            print('----------------------------')

    def add(self, layer):
        if len(self.layers) > 0:          
            self.layers[-1].next_layer = layer
            layer.prev_layer = self.layers[-1]
            layer.initialize()
        self.layers.append(layer)

    def infer(self, sample):
        """ Inference """
        z = sample
        for l in self.layers[1:]:
            z = l.infer(z)
        return z

    def train_batch(self, batch_samples, batch_labels, eta, loss_func):
        """ Single step (batched) SGD """

        b = batch_samples.shape[-1]  # Batch size = number of columns in the batch
        if self.comm != None:
            requests = [MPI.REQUEST_NULL  for i in range(0,len(self.layers))]
            WB  = [None for l in range(0, len(self.layers))]
            aux = [None for l in range(0, len(self.layers))]

        # Forward pass (FP)
        self.layers[0].a = batch_samples
        for l in range(1, len(self.layers)):
            self.layers[l].forward(self.layers[l-1].a)

        total_loss = np.zeros(1)
        loss= np.array([loss_func(batch_labels, self.layers[-1].a)])
        if self.comm != None:
           loss_req = self.comm.Iallreduce( loss, total_loss, op = MPI.SUM)

        # Back propagation. Gradient computation (GC) and calculate changes local
        self.layers[-1].backward((self.layers[-1].a - batch_labels))
        self.layers[-1].calculate_change(b)
        if self.comm != None and len(self.layers[-1].changeW)>0:
            WB[-1] = np.append(self.layers[-1].changeW.reshape(-1), self.layers[-1].changeB.reshape(-1))
            aux[-1] = np.zeros_like(WB[-1])
            requests[-1] = self.comm.Iallreduce( WB[-1], aux[-1], op = MPI.SUM)

    
        for l in range(len(self.layers)-2, 0, -1):
            self.layers[l].backward()
            self.layers[l].calculate_change(b)
            if self.comm != None and len(self.layers[l].changeW)>0:
                WB[l] = np.append(self.layers[l].changeW.reshape(-1), self.layers[l].changeB.reshape(-1))
                aux[l] = np.zeros_like(WB[l])
                requests[l] = self.comm.Iallreduce( WB[l], aux[l], op = MPI.SUM)


        # Weight update (WU)
        for l in range(len(self.layers)-1, 0, -1):
            if self.comm != None and len(self.layers[l].changeW)>0:
                requests[l].Wait()
                self.layers[l].changeW = aux[l][0:self.layers[l].weights.size].reshape(self.layers[l].weights.shape) 
                self.layers[l].changeB = aux[l][WB[l].size-self.layers[l].bias.size:].reshape(self.layers[l].bias.shape)   
            self.layers[l].update_weights(eta, b)

        if self.comm != None:
           loss_req.Wait()
        return total_loss[0]/self.nprocs

    def train(self, samples, labels, eta, nepochs, b, loss_func= "loss", early_stop= True):
        """ SGD over all samples, in batches of size b """

        nsamples = samples.shape[-1] # Numer of samples
        if self.rank == 0: #self.comm == None or (self.comm != None and rank == 0):
            savecost = []                # Error after each epoch training
        loss_func_= getattr(NN_utils, loss_func)
       
        #EPOCHS
        for counter in range(nepochs):

            if self.rank == 0:  
                print('------- Epoch',counter+1)
            s = list(range(nsamples))
            random.shuffle(s)         # shuffle for random ordering of the batch
            counter3 = 1 
            batchGlobal = b*self.nprocs 
            rest = nsamples % batchGlobal
            if rest < (batchGlobal/4): #the last batch do its samples + rest of samples
                endFor = nsamples - batchGlobal - rest
                lastIter = batchGlobal + rest
            else:
                endFor = nsamples - rest  #the rest of samples are a new batch (mini)
                lastIter = rest

            #BATCHS (except the last one)
            for counter2 in range(0, endFor, batchGlobal):
                indices = s[counter2+b*self.rank:counter2+b*(self.rank+1)]  
                batch_samples = samples[...,indices]    # Current batch samples
                batch_labels  = labels[...,indices]     # Current batch labels

                total_loss = self.train_batch(batch_samples, batch_labels, eta/batchGlobal, loss_func_)  #TRAIN

                if self.rank == 0:                                              #TEST
                    #savecost.append(loss_func_(labels, self.infer(samples)))
                    #print('            Batch', counter3, "Cost fnct (%s): " % loss_func, savecost[-1])
                    print('            Batch', counter3, "Cost fnct (%s): " % loss_func, total_loss)
                counter3 = counter3 + 1


           #LAST BATCH
            newb = lastIter // self.nprocs #new size of batch/process
            ini = endFor + newb*self.rank
            end = ini + newb
            if self.rank == self.nprocs - 1:
                end = nsamples #if distribution is not exact, last process do the rest of samples

            indices = s[ini:end]
            batch_samples = samples[...,indices]    # Current batch samples
            batch_labels  = labels[...,indices]     # Current batch labels

            total_loss = self.train_batch(batch_samples, batch_labels, eta/lastIter, loss_func_)     #TRAIN

            if self.rank == 0:                                              #TEST
                #savecost.append(loss_func_(labels, self.infer(samples)))
                #print('            Batch', counter3, "Cost fnct (%s): " % loss_func, savecost[-1])
                print('            Batch', counter3, "Cost fnct (%s): " % loss_func, total_loss)


        print('**** Access order to samples during training')
