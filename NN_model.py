import numpy as np
import math
import random
import ctypes, os
import NN_utils
import pyextrae.common.extrae as pyextrae

from NN_utils import PYDL_EVT, PYDL_OPS_EVT, PYDL_NUM_EVTS, PYDL_OPS_NUM_EVTS
from mpi4py import MPI

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
        layer.id = len(self.layers) - 1

    def define_event_type(self):
        nvalues = len(self.layers) * PYDL_NUM_EVTS + 1
        description = "Model layers"
        values = (ctypes.c_ulonglong * nvalues)()
        description_values = (ctypes.c_char_p * nvalues)()
        values[0] = 0
        description_values[0] = "End".encode('utf-8')
        for i in range(1, nvalues):
          values[i] = i
        for i in range(len(self.layers)):
          description_values[i*PYDL_NUM_EVTS+1] = (str(i) + "_" + type(self.layers[i]).__name__ + "_inference ").encode('utf-8')
          description_values[i*PYDL_NUM_EVTS+2] = (str(i) + "_" + type(self.layers[i]).__name__ + "_forward ").encode('utf-8')
          description_values[i*PYDL_NUM_EVTS+3] = (str(i) + "_" + type(self.layers[i]).__name__ + "_compute_dX ").encode('utf-8')
          description_values[i*PYDL_NUM_EVTS+4] = (str(i) + "_" + type(self.layers[i]).__name__ + "_compute_dW ").encode('utf-8')
          description_values[i*PYDL_NUM_EVTS+5] = (str(i) + "_" + type(self.layers[i]).__name__ + "_allreduce_dW ").encode('utf-8')
          description_values[i*PYDL_NUM_EVTS+6] = (str(i) + "_" + type(self.layers[i]).__name__ + "_wait_dW ").encode('utf-8')
          description_values[i*PYDL_NUM_EVTS+7] = (str(i) + "_" + type(self.layers[i]).__name__ + "_update_dW ").encode('utf-8')

        pyextrae.Extrae[os.getpid()].Extrae_define_event_type(
            ctypes.pointer(ctypes.c_uint(NN_utils.PYDL_EVT)),
            ctypes.c_char_p(description.encode('utf-8')),
            ctypes.pointer(ctypes.c_uint(nvalues)),
            ctypes.pointer(values),
            ctypes.pointer(description_values) )

        nvalues = len(self.layers) * PYDL_OPS_NUM_EVTS + 1
        description = "PYDL ops per layer"
        values = (ctypes.c_ulonglong * nvalues)()
        description_values = (ctypes.c_char_p * nvalues)()
        values[0] = 0
        description_values[0] = "End".encode('utf-8')
        for i in range(1, nvalues):
          values[i] = i
        for i in range(len(self.layers)):
          description_values[i*PYDL_OPS_NUM_EVTS+1] = (str(i) + "_" + type(self.layers[i]).__name__ + "_inference_im2col ").encode('utf-8')
          description_values[i*PYDL_OPS_NUM_EVTS+2] = (str(i) + "_" + type(self.layers[i]).__name__ + "_inference_matmul ").encode('utf-8')
          description_values[i*PYDL_OPS_NUM_EVTS+3] = (str(i) + "_" + type(self.layers[i]).__name__ + "_forward_im2col ").encode('utf-8')
          description_values[i*PYDL_OPS_NUM_EVTS+4] = (str(i) + "_" + type(self.layers[i]).__name__ + "_forward_matmul ").encode('utf-8')
          description_values[i*PYDL_OPS_NUM_EVTS+5] = (str(i) + "_" + type(self.layers[i]).__name__ + "_compute_dX_im2col ").encode('utf-8')
          description_values[i*PYDL_OPS_NUM_EVTS+6] = (str(i) + "_" + type(self.layers[i]).__name__ + "_compute_dX_matmul ").encode('utf-8')
          description_values[i*PYDL_OPS_NUM_EVTS+7] = (str(i) + "_" + type(self.layers[i]).__name__ + "_compute_dW_im2col ").encode('utf-8')
          description_values[i*PYDL_OPS_NUM_EVTS+8] = (str(i) + "_" + type(self.layers[i]).__name__ + "_compute_dW_matmul ").encode('utf-8')
          description_values[i*PYDL_OPS_NUM_EVTS+9] = (str(i) + "_" + type(self.layers[i]).__name__ + "_wait_allreduce_dW ").encode('utf-8')

        pyextrae.Extrae[os.getpid()].Extrae_define_event_type(
            ctypes.pointer(ctypes.c_uint(NN_utils.PYDL_OPS_EVT)),
            ctypes.c_char_p(description.encode('utf-8')),
            ctypes.pointer(ctypes.c_uint(nvalues)),
            ctypes.pointer(values),
            ctypes.pointer(description_values) )

    def inference(self, sample):
        """ Inference """
        z = sample
        for l in self.layers[1:]:
            pyextrae.eventandcounters(PYDL_EVT, l.id * 7 + 1)
            z = l.infer(z)
            pyextrae.eventandcounters(PYDL_EVT, 0)
        return z

    def train_batch(self, batch_samples, batch_labels, eta, loss_func):
        """ Single step (batched) SGD """

        b = batch_samples.shape[-1]  # Batch size = number of columns in the batch

        # Forward pass (FP)
        self.layers[0].a = batch_samples
        for l in range(1, len(self.layers)):
            pyextrae.eventandcounters(PYDL_EVT, self.layers[l].id * 7 + 2)
            self.layers[l].forward(self.layers[l-1].a)
            pyextrae.eventandcounters(PYDL_EVT, 0)

        total_loss = np.zeros(1)
        loss= np.array([loss_func(batch_labels, self.layers[-1].a)])
        if self.comm != None:
           loss_req = self.comm.Iallreduce( loss, total_loss, op = MPI.SUM)

        # Back propagation. Gradient computation (GC) and calculate changes local
        for l in range(len(self.layers)-1, 0, -1):
            pyextrae.eventandcounters(PYDL_EVT, self.layers[l].id * PYDL_NUM_EVTS + 3)
            if l == len(self.layers)-1: dX = (self.layers[-1].a - batch_labels)
            else:                       dX = []
            self.layers[l].backward(dX)
            pyextrae.eventandcounters(PYDL_EVT, 0)

            pyextrae.eventandcounters(PYDL_EVT, self.layers[l].id * PYDL_NUM_EVTS + 4)
            self.layers[l].calculate_change(b)
            pyextrae.eventandcounters(PYDL_EVT, 0)

            pyextrae.eventandcounters(PYDL_EVT, self.layers[l].id * PYDL_NUM_EVTS + 5)
            self.layers[l].reduce_weights(self.comm)
            pyextrae.eventandcounters(PYDL_EVT, 0)

        # Weight update (WU)
        for l in range(len(self.layers)-1, 0, -1):
            pyextrae.neventandcounters([PYDL_EVT, PYDL_OPS_EVT], [self.layers[l].id * PYDL_NUM_EVTS + 6, self.layers[l].id * PYDL_OPS_NUM_EVTS + 9])
            self.layers[l].wait_allreduce(self.comm)
            pyextrae.neventandcounters([PYDL_EVT, PYDL_OPS_EVT], [0, 0])

            pyextrae.eventandcounters(PYDL_EVT, self.layers[l].id * PYDL_NUM_EVTS + 7)
            self.layers[l].update_weights(eta, b)
            pyextrae.eventandcounters(PYDL_EVT, 0)

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


        #print('**** Access order to samples during training')
        self.define_event_type()
