#!/usr/bin/python

from __future__ import print_function

TracingLibrary = "libmpitrace.so"
import ctypes
ctypes.CDLL("/home/dolzm/install/extrae-3.6.0/lib/" + TracingLibrary)
import pyextrae.common.extrae as pyextrae
pyextrae.startTracing( TracingLibrary )

from mpi4py import MPI
import random
import numpy
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from NN_model import *
from NN_layer import *
from models import *
from datasets import *

def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='simplecnn')
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--steps_per_epoch', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--parallel', type=str, default=None)
    parser.add_argument('--inference', action="store_true")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_options()

    if args.parallel == "data":
        comm = MPI.COMM_WORLD
        batch_factor = 1
        nprocs = comm.Get_size()
        rank = comm.Get_rank()

    elif args.parallel == None:
        comm = None
        batch_factor = 1
        nprocs = 1
        rank = 0
    
    # A couple of details...
    random.seed(0)
    numpy.set_printoptions(precision=15)
    numpy.random.seed(30)
    
    model = create_model(args.model, comm)
    x, y  = read_dataset(args.dataset)

    if args.model == "vgg16":
        x = x[:224,:224,...] 

    if rank == 0:
        print('Running with %d procs' % nprocs)
        print('**** Creating %s model...', args.model)
        model.show()
    
    #eta     = 0.1               # Learning rate
    #nepochs = 1                 # Number of epochs to train
    #steps   = 6
    b       = args.batch_size * batch_factor # Batch size
    
    if args.steps_per_epoch != 0:
       subset_size = b * nprocs * args.steps_per_epoch
       x = x[...,:subset_size]
       y = y[...,:subset_size]
    
    if rank == 0:
        if args.inference:
            targ= np.argmax(y, axis=0)
            pred= np.argmax(model.inference(x), axis=0)
            print("Accuracy: %.2f %%" % (np.sum(np.equal(targ, pred))*100/targ.shape[0]))
            print(np.sum(np.equal(targ, pred)), targ.shape[0])
    
        print('**** Training...')
        print('     Epochs:', args.num_epochs, 'Batch size:', b, 'Learning rate:', args.learning_rate)
        t1 = time.time()
    
    if args.parallel:
        comm.Barrier()

    model.train(x, y, args.learning_rate, args.num_epochs, b, loss_func="accuracy")
    
    if rank == 0:
        t2 = time.time()
        print('**** Done... and thanks for all the fish!!!')
        print('**** Time: ', t2-t1)
        
        if args.inference:
            targ= np.argmax(y, axis=0)
            pred= np.argmax(model.inference(x), axis=0)
            print("Accuracy: %.2f %%" % (np.sum(np.equal(targ, pred))*100/targ.shape[0]))
            print(np.sum(np.equal(targ, pred)), targ.shape[0])
