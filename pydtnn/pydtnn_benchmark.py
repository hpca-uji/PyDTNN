#!/usr/bin/env python

"""
PyDTNN Benchmark script
"""

#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021 Universitat Jaume I
#
#  PyDTNN is free software: you can redistribute it and/or modify it under the
#  terms of the GNU General Public License as published by the Free Software
#  Foundation, either version 3 of the License, or (at your option) any later
#  version.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT
#  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
#  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
#  License for more details.
#
#  You should have received a copy of the GNU General Public License along
#  with this program. If not, see <https://www.gnu.org/licenses/>.

# from __future__ import print_function

import cProfile
import os
import pstats
import random
import subprocess
import sys
import time
from io import StringIO

import numpy as np

from pydtnn.parser import parser
from pydtnn.datasets import get_dataset
from pydtnn.model import Model
from pydtnn.optimizers import get_optimizer
from pydtnn.lr_schedulers import get_lr_schedulers


Extrae_tracing = False
if os.environ.get("EXTRAE_ON", None) == "1":
    TracingLibrary = "libptmpitrace.so"
    import pyextrae.common.extrae as pyextrae

    pyextrae.startTracing(TracingLibrary)
    Extrae_tracing = True

def show_options(params):
    for arg in vars(params):
        if arg != "comm":
            print(f'  {arg:31s}: {str(getattr(params, arg)):s}')
            # print(f'  --{arg:s}={str(getattr(params, arg)):s} \\')

def main():
    # Parse options
    params = parser.parse_args()
    # Adjust params based on the given command line arguments
    _MPI = None
    if params.parallel in ["data"]:
        from mpi4py import MPI
        _MPI = MPI
        params.comm = MPI.COMM_WORLD
        params.mpi_processes = params.comm.Get_size()
        rank = params.comm.Get_rank()
        params.global_batch_size = params.batch_size * params.mpi_processes
        if params.optimizer_name == "sgd" and params.learning_rate_scaling:
            params.learning_rate *= params.mpi_processes
    elif params.parallel == "sequential":
        params.comm = None
        params.mpi_processes = 1
        rank = 0
    else:
        raise ValueError(f"Parallel option '{params.parallel}' not recognized.")
    params.threads_per_process = os.environ.get("OMP_NUM_THREADS", 1)
    try:
        params.gpus_per_node = subprocess.check_output(["nvidia-smi", "-L"]).count(b'UUID')
    except (FileNotFoundError, subprocess.CalledProcessError):
        params.gpus_per_node = 0
    if params.enable_gpu and params.parallel == "data":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % params.gpus_per_node)

    # Initialize random seeds to 0
    random.seed(0)
    np.random.seed(0)
    # Create model
    model = Model(**vars(params))
    # Print model
    if rank == 0:
        print(f'**** {model.model_name} model...')
        model.show()
        print(f'**** Loading {model.dataset_name} dataset...')
    # Load weights and bias
    if model.weights_and_bias_filename:
        model.load_weights_and_bias(model.weights_and_bias_filename)
    # Metrics
    metrics_list = [f for f in model.metrics.replace(" ", "").split(",")]
    # Dataset
    dataset = get_dataset(model)
    if model.steps_per_epoch > 0:
        dataset.adjust_steps_per_epoch(model.steps_per_epoch,
                                       model.batch_size, model.mpi_processes)
    # Optimizers and LRSchedulers
    optimizer = get_optimizer(model)
    lr_schedulers = get_lr_schedulers(model)
    # Print parameters
    if rank == 0:
        print('**** Parameters:')
        show_options(params)
    # First (or unique) evaluation
    if model.evaluate or model.evaluate_only:
        if rank == 0:
            print('**** Evaluating on test dataset...')
            t1 = time.time()
        _ = model.evaluate_dataset(dataset, model.batch_size, model.loss_func, metrics_list)
        if rank == 0:
            t2 = time.time()
            total_time = t2 - t1
            print(f'Testing time: {total_time:5.4f} s')
            print(f'Testing throughput: {dataset.test_nsamples / total_time:5.4f} samples/s')
            print(f'Testing time (from model): {model.perf_counter.testing_time:5.4f} s')
            print(f'Testing throughput (from model): {model.perf_counter.testing_throughput:5.4f} samples/s')
            print(f'Testing maximum memory allocated: ',
                  f'{model.perf_counter.testing_maximum_memory / 1024:.2f} MiB')
            print(f'Testing mean memory allocated: ',
                  f'{model.perf_counter.testing_mean_memory / 1024:.2f} MiB')
        if model.evaluate_only:
            sys.exit(0)
    # Barrier
    if model.parallel in ["data"]:
        model.comm.Barrier()
    # Training
    if rank == 0:
        # print('**** Model time: ', model.calculate_time())
        print('**** Training...')
        t1 = time.time()
        if model.profile:
            pr = cProfile.Profile()
            pr.enable()
    # Training a model directly from a dataset
    history = model.train_dataset(dataset,
                                  nepochs = model.num_epochs,
                                  local_batch_size = model.batch_size,
                                  val_split = model.validation_split,
                                  loss = model.loss_func,
                                  metrics_list = metrics_list,
                                  optimizer = optimizer,
                                  lr_schedulers = lr_schedulers)
    # Alternatively, the model can be trained on any specific data
    # history = model.train(x_train=dataset.X_train_val, y_train=dataset.Y_train_val,
    #                       x_val=dataset.x_test, y_val=dataset.y_test,
    #                       nepochs=params.num_epochs,
    #                       local_batch_size=params.batch_size,
    #                       loss=params.loss_func,
    #                       metrics_list=metrics_list,
    #                       optimizer=optimizer,
    #                       lr_schedulers=lr_schedulers)
    # Barrier
    if model.parallel in ["data"]:
        model.comm.Barrier()
    # Print performance results and evaluation history
    if rank == 0:
        if model.profile:
            pr.disable()
            s = StringIO()
            sortby = 'time'
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print(s.getvalue())
        t2 = time.time()
        print('**** Done...')
        total_time = t2 - t1
        print(f'Training time: {total_time:5.4f} s')
        if model.perf_counter.num_epochs > 0:
            print(f'Time per epoch: {total_time / model.perf_counter.num_epochs:5.4f} s')
            print(f'Training throughput: '
                  f'{(dataset.train_val_nsamples * model.perf_counter.num_epochs) / total_time:5.4f} samples/s')
            print(f'Training time (from model): {model.perf_counter.training_time:5.4f} s')
            print(f'Training time per epoch (from model): '
                  f'{model.perf_counter.training_time / model.perf_counter.num_epochs:5.4f} s')
            print(f'Training throughput (from model): {model.perf_counter.training_throughput:5.4f} samples/s')
            print(f'Training time (from model, estimated from last half of each epoch): '
                  f'{model.perf_counter.training_time_estimated_from_last_half_of_each_epoch:5.4f} s')
            print(f'Training throughput (from model, from last half of each epoch): '
                  f'{model.perf_counter.training_throughput_only_last_half_of_each_epoch:5.4f} samples/s')
            print(f'Training maximum memory allocated: '
                  f'{model.perf_counter.training_maximum_memory / 1024:.2f} MiB')
            print(f'Training mean memory allocated: '
                  f'{model.perf_counter.training_mean_memory / 1024:.2f} MiB')
        if model.history_file:
            with open(model.history_file, "w") as f:
                keys = [k for k in history]
                for v in range(len(history[keys[0]])):
                    f.write(' '.join(["%3d" % v] +
                                     [('%20.4f' % history[k][v]) for k in keys]) + '\n')
    # Second (and last) evaluation
    if model.evaluate:
        if rank == 0:
            print('**** Evaluating on test dataset...')
        _ = model.evaluate_dataset(dataset, model.batch_size, model.loss_func, metrics_list)
    # Barrier and finalize
    if model.comm is not None and _MPI is not None:
        model.comm.Barrier()
        # The next line is required if running under SLURM (it seems it is not automatically called at exit)
        _MPI.Finalize()


if __name__ == "__main__":
    main()
