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
the presence of multicore processors and GPUs at node level. For that, PyDTNN 
uses MPI4Py for message-passing, BLAS calls via NumPy for multicore processors
and PyCUDA+cuDNN+cuBLAS for NVIDIA GPUs.

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
__version__ = "1.1.0"

import os

Extrae_tracing = False
if "EXTRAE_ON" in os.environ and os.environ["EXTRAE_ON"] == "1":
  TracingLibrary = "libptmpitrace.so"
  import ctypes
  ctypes.CDLL("/mnt/beegfs/users/dolzm/install/extrae-3.6.0/lib/" + TracingLibrary)

  import pyextrae.common.extrae as pyextrae
  pyextrae.startTracing( TracingLibrary )
  Extrae_tracing = True
  
import numpy, os, sys, math, time, argparse, subprocess, numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def parse_options():
    bool_lambda = lambda x: (str(x).lower() in ['true','1', 'yes'])
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--model', type=str, default="simplecnn")
    parser.add_argument('--dataset', type=str, default="mnist")
    parser.add_argument('--dataset_train_path', type=str, default="../datasets/mnist")
    parser.add_argument('--dataset_test_path', type=str, default="../datasets/mnist")
    parser.add_argument('--test_as_validation', default=False, type=bool_lambda)
    parser.add_argument('--flip_images', default=False, type=bool_lambda)
    parser.add_argument('--flip_images_prob', type=float, default=0.5)
    parser.add_argument('--crop_images', default=False, type=bool_lambda)
    parser.add_argument('--crop_images_size', type=int, default=16)
    parser.add_argument('--crop_images_prob', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--global_batch_size', type=int, default=None)
    parser.add_argument('--validation_split', type=float, default=0.0)
    parser.add_argument('--steps_per_epoch', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--evaluate', default=False, type=bool_lambda)
    parser.add_argument('--weights_and_bias_filename',  type=str, default=None)
    parser.add_argument('--shared_storage', default=False, type=bool_lambda)
    parser.add_argument('--history_file', type=str, default=None)
    # Optimizer
    parser.add_argument('--optimizer', type=str, default="sgd")
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--learning_rate_scaling', default=True, type=bool_lambda)
    parser.add_argument('--momentum', type=float, default=0.9)    
    parser.add_argument('--decay', type=float, default=0.0)
    parser.add_argument('--nesterov', default=False, type=bool_lambda)
    parser.add_argument('--beta1', type=float, default=0.99)
    parser.add_argument('--beta2', type=float, default=0.999)    
    parser.add_argument('--epsilon', type=float, default=1e-7)
    parser.add_argument('--rho', type=float, default=0.9)
    parser.add_argument('--loss_func', type=str, default="categorical_cross_entropy")
    parser.add_argument('--metrics', type=str, default="categorical_accuracy")
    # Learning rate schedulers
    parser.add_argument('--lr_schedulers', type=str, default="early_stopping,reduce_lr_on_plateau,model_checkpoint")
    parser.add_argument('--warm_up_epochs', type=int, default=5)
    parser.add_argument('--early_stopping_metric', type=str, default="val_categorical_cross_entropy")
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    parser.add_argument('--reduce_lr_on_plateau_metric', type=str, default="val_categorical_cross_entropy")
    parser.add_argument('--reduce_lr_on_plateau_factor', type=float, default=0.1)
    parser.add_argument('--reduce_lr_on_plateau_patience', type=int, default=5)
    parser.add_argument('--reduce_lr_on_plateau_min_lr', type=float, default=0)
    parser.add_argument('--reduce_lr_every_nepochs_factor', type=float, default=0.1)
    parser.add_argument('--reduce_lr_every_nepochs_nepochs', type=int, default=5)
    parser.add_argument('--reduce_lr_every_nepochs_min_lr', type=float, default=0)  
    parser.add_argument('--stop_at_loss_metric', type=str, default="val_accuracy")
    parser.add_argument('--stop_at_loss_threshold', type=float, default=0)      
    parser.add_argument('--model_checkpoint_metric', type=str, default="val_categorical_cross_entropy")
    parser.add_argument('--model_checkpoint_save_freq', type=int, default=2)
    # Parallelization + tracing
    parser.add_argument('--mpi_processes', type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument('--threads_per_process', type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument('--parallel', type=str, default="sequential")
    parser.add_argument('--non_blocking_mpi', type=bool_lambda,  default=False)
    parser.add_argument('--tracing', type=bool_lambda, default=False)
    parser.add_argument('--profile', type=bool_lambda, default=False)
    parser.add_argument('--gpus_per_node', type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument('--enable_gpu', type=bool_lambda, default=False)
    parser.add_argument('--enable_gpudirect', type=bool_lambda, default=False)
    parser.add_argument('--enable_nccl', type=bool_lambda, default=False)
    parser.add_argument('--dtype', type=str, default="float32")
    parser.add_argument('--cpu_speed', type=float, default=4e12, help=argparse.SUPPRESS)
    parser.add_argument('--memory_bw', type=float, default=50e9, help=argparse.SUPPRESS)
    parser.add_argument('--network_bw', type=float, default=1e9, help=argparse.SUPPRESS)
    parser.add_argument('--network_lat', type=float, default=0.5e-6, help=argparse.SUPPRESS)
    parser.add_argument('--network_alg', type=str, default="vdg", help=argparse.SUPPRESS)
    return parser.parse_args()

def show_options(params):
    for arg in vars(params):
        if arg != "comm":
            print(f'  {arg:31s}: {str(getattr(params, arg)):s}')
            #print(f'  --{arg:s}={str(getattr(params, arg)):s} \\')

def get_optimizer(params):
    opt_mod = importlib.import_module("NN_optimizer%s" % ("","_gpu")[params.enable_gpu])
    opt_ref = getattr(opt_mod, "%s%s" % (params.optimizer, ("","_gpu")[params.enable_gpu]))
    if params.optimizer == "rmsprop":
        opt = opt_ref(learning_rate = params.learning_rate,
                  rho = params.rho,
                  epsilon = params.epsilon,
                  decay = params.decay,
                  dtype = params.dtype)
    elif params.optimizer == "adam":
        opt = opt_ref(learning_rate = params.learning_rate,
                  beta1 = params.beta1,
                  beta2 = params.beta2, 
                  epsilon = params.epsilon,
                  decay = params.decay,
                  dtype = params.dtype)
    elif params.optimizer == "nadam":
        opt = opt_ref(learning_rate = params.learning_rate,
                  beta1 = params.beta1,
                  beta2 = params.beta2,
                  epsilon = params.epsilon,
                  decay = params.decay,
                  dtype = params.dtype)
    else: # "sgd":
        opt = opt_ref(learning_rate = params.learning_rate,
                  momentum = params.momentum,
                  nesterov = params.nesterov,
                  decay = params.decay,        
                  dtype = params.dtype)
    return opt

def get_lr_schedulers(params):
    lr_schedulers = []
    for lr_sched in params.lr_schedulers.split(","):
        if lr_sched == "warm_up":
            lrs = WarmUpLRScheduler(params.warm_up_epochs, 
                  params.learning_rate / params.mpi_processes,
                  params.learning_rate)
        elif lr_sched == "early_stopping":
            lrs = EarlyStopping(params.early_stopping_metric, 
                  params.early_stopping_patience)
        elif lr_sched == "reduce_lr_on_plateau":
            lrs = ReduceLROnPlateau(params.reduce_lr_on_plateau_metric, 
                  params.reduce_lr_on_plateau_factor,
                  params.reduce_lr_on_plateau_patience,
                  params.reduce_lr_on_plateau_min_lr)
        elif lr_sched == "reduce_lr_every_nepochs":
            lrs = ReduceLREveryNEpochs(params.reduce_lr_every_nepochs_factor,
                  params.reduce_lr_every_nepochs_nepochs,
                  params.reduce_lr_every_nepochs_min_lr)
        elif lr_sched == "stop_at_loss":
            lrs = StopAtLoss(params.stop_at_loss_metric,
                  params.stop_at_loss_threshold)
        elif lr_sched == "model_checkpoint":
            lrs = ModelCheckpoint(params.model_checkpoint_metric, 
                  params.model_checkpoint_save_freq)            
        else: lrs = None
        if lrs: lr_schedulers.append(lrs)
    return lr_schedulers

if __name__ == "__main__":
    params = parse_options()
    params.dtype = getattr(np, params.dtype)

    if params.parallel in ["data", "hybrid"]:
        from mpi4py import MPI
        params.comm = MPI.COMM_WORLD
        params.mpi_processes = params.comm.Get_size()
        rank = params.comm.Get_rank()
        params.global_batch_size = params.batch_size * params.mpi_processes
        if params.optimizer == "sgd" and params.learning_rate_scaling:
            params.learning_rate *= params.mpi_processes

    elif params.parallel == "sequential":
        params.comm = None
        params.mpi_processes = 1
        rank = 0
    
    if "OMP_NUM_THREADS" in os.environ:
        params.threads_per_process = os.environ["OMP_NUM_THREADS"]
    else:
        params.threads_per_process = 1

    try:
        params.gpus_per_node = str(subprocess.check_output(["nvidia-smi", "-L"])).count('UUID')
    except:
        params.gpus_per_node = 0

    if params.enable_gpu and params.parallel == "data":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % params.gpus_per_node)

    from NN_model import *
    from NN_layer import *
    from NN_optimizer import *
    from NN_lr_scheduler import *
    from models import *
    from datasets import *

    # A couple of details...
    random.seed(0)
    numpy.random.seed(0)
    # numpy.set_printoptions(precision=15)

    model = get_model(params)
    if rank == 0:
        print('**** %s model...' % params.model)
        model.show()    
        print('**** Loading %s dataset...' % params.dataset)

    if params.weights_and_bias_filename:
        model.load_weights_and_bias(params.weights_and_bias_filename)

    metrics = [f for f in params.metrics.replace(" ","").split(",")]

    dataset = get_dataset(params)
    if params.steps_per_epoch > 0:
        dataset.adjust_steps_per_epoch(params.steps_per_epoch, 
                                       params.batch_size, params.mpi_processes)

    optimizer = get_optimizer(params)
    lr_schedulers = get_lr_schedulers(params)

    if rank == 0:
        print('**** Parameters:')
        show_options(params)

    if params.evaluate:
        if rank == 0:
            print('**** Evaluating on test dataset...')        
        test_loss = model.evaluate_dataset(dataset, params.batch_size, \
                                           params.loss_func, metrics)
  
    if params.parallel in ["data", "model"]:
        params.comm.Barrier()

    if rank == 0:
        # print('**** Model time: ', model.calculate_time())
        print('**** Training...')
        t1 = time.time()

        if params.profile:
            import cProfile, pstats
            from io import StringIO
            pr = cProfile.Profile(); pr.enable()

    # Training a model directly from a dataset
    history = model.train_dataset(dataset,
                        nepochs          = params.num_epochs, 
                        local_batch_size = params.batch_size,
                        val_split        = params.validation_split,  
                        loss             = params.loss_func,
                        metrics          = metrics,
                        optimizer        = optimizer,
                        lr_schedulers    = lr_schedulers)

    # Alternatively, the model can be trained on any specific data
    # history = model.train(X_train = dataset.X_train_val, Y_train = dataset.Y_train_val,
    #             X_val   = dataset.X_test,      Y_val   = dataset.Y_test,
    #             nepochs          = params.num_epochs, 
    #             local_batch_size = params.batch_size,
    #             loss_metrics     = loss_metrics, 
    #             optimizer        = optimizer
    #             lr_schedulers    = lr_schedulers)

    if params.parallel in ["data", "model"]:
        params.comm.Barrier()

    if rank == 0:
        if params.profile:
            pr.disable(); s = StringIO(); sortby = 'time'
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats(); print(s.getvalue())

        t2 = time.time()
        print('**** Done...')
        total_time = (t2-t1)
        print(f'Time: {total_time:5.4f} s')
        print(f'Time per epoch: {total_time / params.num_epochs:5.4f} s')
        print(f'Throughput: {(dataset.train_val_nsamples * params.num_epochs)/total_time:5.4f} samples/s')

        if params.history_file:
            with open(params.history_file, "w") as f:
                keys = [k for k in history]
                for v in range(len(history[keys[0]])):
                    f.write(' '.join(["%3d" % v] + \
                        [('%20.4f' % history[k][v]) for k in keys]) + '\n')

    if params.evaluate:
        if rank == 0:
            print('**** Evaluating on test dataset...')
        test_loss = model.evaluate_dataset(dataset, params.batch_size, \
                                           params.loss_func, metrics)
