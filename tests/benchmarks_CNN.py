#!/usr/bin/python
from __future__ import print_function

""" Python Distributed Training of Neural Networks - PyDTNN

PyDTNN is a light-weight library for distributed Deep Learning training and 
inference that offers an initial starting point for interaction with 
distributed training of (and inference with) deep neural networks. PyDTNN 
priorizes simplicity over efficiency, providing an amiable user interface 
which enables a flat accessing curve. To perform the training and inference 
processes, PyDTNN exploits distributed inter-process parallelism (via MPI) 
for clusters and intra-process (via multi-threading) parallelism to leverage 
the presence of multicore processors at node level.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.

"""

__author__ = "Manuel F. Dolz, Enrique S. Quintana, \
              Mar Catalan, Adrian Castello"
__contact__ = "dolzm@uji.es"
__copyright__ = "Copyright 2020, Universitat Jaume I"
__credits__ = ["Manuel F. Dolz, Enrique S. Quintana", \
               "Mar Catalan", "Adrian Castello"]
__date__ = "2020/03/22"

__email__ =  "dolzm@uji.es"
__license__ = "GPLv3"
__maintainer__ = "Manuel F. Dolz"
__status__ = "Production"
__version__ = "1.0.0"

import os

Extrae_tracing = False
if "EXTRAE_ON" in os.environ and os.environ["EXTRAE_ON"] == 1:
  TracingLibrary = "libptmpitrace.so"
  import ctypes
  ctypes.CDLL("/home/dolzm/install/extrae-3.6.0/lib/" + TracingLibrary)

  import pyextrae.common.extrae as pyextrae
  pyextrae.startTracing( TracingLibrary )
  Extrae_tracing = True
  
import numpy, os, sys, math, time, argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from NN_model import *
from NN_layer import *
from models import *
from datasets import *

def parse_options():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--model', type=str, default="simplecnn")
    parser.add_argument('--dataset', type=str, default="mnist")
    parser.add_argument('--dataset_train_path', type=str, default="../datasets/mnist")
    parser.add_argument('--dataset_test_path', type=str, default="../datasets/mnist")
    parser.add_argument('--test_as_validation', action="store_true", default=False)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--validation_split', type=float, default=0.0)
    parser.add_argument('--steps_per_epoch', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--evaluate', action="store_true", default=False)
    # Optimizer
    parser.add_argument('--optimizer', type=str, default="SGDMomentum")
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--decay_rate', type=float, default=0.99)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--loss_func', type=str, default="accuracy,categorical_cross_entropy")
    # Parallelization + tracing
    parser.add_argument('--parallel', type=str, default="sequential")
    parser.add_argument('--non_blocking_mpi', action="store_true", default=False)
    parser.add_argument('--tracing', action="store_true", default=False)
    parser.add_argument('--profile', action="store_true", default=False)
    parser.add_argument('--enable_gpu', action="store_true", default=False)
    parser.add_argument('--dtype', type=str, default="float32")

    return parser.parse_args()

def show_options(params):
    for arg in vars(params):
        if arg != "comm":
            print(f'  {arg:19s}: {str(getattr(params, arg)):s}')
            #print(f'  --{arg:s}={str(getattr(params, arg)):s} \\')

if __name__ == "__main__":
    params = parse_options()

    if params.parallel in ["data", "hybrid"]:
        from mpi4py import MPI
        params.comm = MPI.COMM_WORLD
        nprocs = params.comm.Get_size()
        rank = params.comm.Get_rank()
        if rank == 0:
            print('**** Running with %d processes...' % nprocs)

    elif params.parallel == "sequential":
        params.comm = None
        nprocs = 1
        rank = 0
    
    # A couple of details...
    random.seed(0)
    numpy.random.seed(0)
    numpy.set_printoptions(precision=15)

    model = get_model(params)

    if rank == 0:
        print('**** Creating %s model...' % params.model)
        model.show()
        print('**** Parameters:')
        show_options(params)        
        print('**** Loading %s dataset...' % params.dataset)

    dataset = get_dataset(params)

    loss_metrics = [f for f in params.loss_func.replace(" ","").split(",")]

    if params.steps_per_epoch > 0:
        dataset.adjust_steps_per_epoch(params.steps_per_epoch, params.batch_size, nprocs)

    if params.evaluate and dataset.X_test.shape[0] > 0:
        if rank == 0:
            print('**** Evaluating on test dataset...')        
        test_loss = model.evaluate(dataset.X_test, dataset.Y_test, loss_metrics)
        if rank == 0:
            print(model.get_metric_results(test_loss, loss_metrics))

    if params.parallel in ["data", "model"]:
        params.comm.Barrier()

    if rank == 0:
        print('**** Training...')
        t1 = time.time()

        if params.profile:
            import cProfile, pstats
            from io import StringIO
            pr = cProfile.Profile(); pr.enable()
    
    # Training a model directly from a dataset
    model.train_dataset(dataset,
                         nepochs                = params.num_epochs, 
                         local_batch_size       = params.batch_size,
                         val_split              = params.validation_split,  
                         loss_metrics           = loss_metrics, 
                         optimizer              = params.optimizer)

    # Alternatively, the model can be trained on any specific data
    # model.train(X_train = dataset.X_train_val, Y_train = dataset.Y_train_val,
    #             X_val   = dataset.X_test,      Y_val   = dataset.Y_test,
    #             nepochs          = params.num_epochs, 
    #             local_batch_size = params.batch_size,
    #             loss_metrics     = loss_metrics, 
    #             optimizer        = params.optimizer)

    if params.parallel in ["data", "model"]:
        params.comm.Barrier()

    if rank == 0:
        if params.profile:
            pr.disable(); s = StringIO(); sortby = 'time'
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats(); print(s.getvalue())

        t2 = time.time()
        print('**** Done... and thanks for all the fish!!!')
        total_time = (t2-t1)
        print(f'Time: {total_time:5.2f} s')
        print(f'Throughput: {(dataset.train_val_nsamples * params.num_epochs)/total_time:5.2f} samples/s')

    if params.evaluate and dataset.X_test.shape[0] > 0:
        if rank == 0:
            print('**** Evaluating on test dataset...')
        test_loss = model.evaluate(dataset.X_test, dataset.Y_test, loss_metrics)
        if rank == 0:
            print(model.get_metric_results(test_loss, loss_metrics))
