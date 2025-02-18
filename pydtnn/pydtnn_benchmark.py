#!/usr/bin/env python

"""
PyDTNN Benchmark script
"""

#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021-22 Universitat Jaume I
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
import sys
import time
from io import StringIO
import platform

import numpy as np

from pydtnn.model import Model
from pydtnn.parser import parser
from pydtnn.utils.best_of import BestOf

Extrae_tracing = False
if os.environ.get("EXTRAE_ON", None) == "1":
    TracingLibrary = "libptmpitrace.so"
    # noinspection PyUnresolvedReferences
    import pyextrae.common.extrae as pyextrae

    pyextrae.startTracing(TracingLibrary)
    Extrae_tracing = True


def show_options(params):
    for arg in vars(params):
        if arg != "comm":
            print(f'  {arg:31s}: {str(getattr(params, arg)):s}')
            # print(f'  --{arg:s}={str(getattr(params, arg)):s} \\')


def print_model_reports(model):
    # Print performance counter report
    model.perf_counter.print_report()
    # Print BestOf report
    if model.enable_best_of:
        print()
        BestOf.print_report()


def main():
    # Parse options
    params = parser.parse_args()
    # Initialize random seeds to 0
    random.seed(0)
    np.random.seed(0)
    # Create model
    model = Model(**vars(params))
    # Gather processes allocated on hosts
    rank_host = f"P{model.rank} allocated on {platform.node()}"
    gathered_rank_host = model.comm.allgather(rank_host)
    # Print model
    if model.rank == 0:
        for rank_host in gathered_rank_host:
            print(rank_host)
        print(f'**** {model.model_name} model...')
        model.show()
    # Print parameters
    if model.rank == 0:
        print('**** Parameters:')
        parser.print_args()
    # First (or unique) evaluation
    if model.evaluate_on_train or model.evaluate_only:
        if model.rank == 0:
            print('**** Evaluating on test dataset...')
            t1 = time.time()
        _ = model.evaluate_dataset()
        if model.rank == 0:
            t2 = time.time()
            # noinspection PyUnboundLocalVariable
            total_time = t2 - t1
            if model.evaluate_only:
                print(f'Testing time: {total_time:5.4f} s')
                print(f'Testing throughput: {model.dataset.test_nsamples / total_time:5.4f} samples/s')
        if model.evaluate_only:
            print_model_reports(model)
            sys.exit(0)
    # Barrier
    if model.parallel in ["data"]:
        model.comm.Barrier()
    # Training
    if model.rank == 0:
        # print('**** Model time: ', model.calculate_time())
        print('**** Training...')
        t1 = time.time()
        if model.profile:
            pr = cProfile.Profile()
            pr.enable()
    # Training a model directly from a dataset
    history = model.train_dataset()
    # Alternatively, the model can be trained on any specific data
    # history = model.train(x_train=x_train, y_train=y_train_val,
    #                       x_val=x_test, y_val=y_test)
    # Barrier
    if model.parallel == "data":
        model.comm.Barrier()
    # Print performance results and evaluation history
    if model.rank == 0:
        if model.profile:
            # noinspection PyUnboundLocalVariable
            pr.disable()
            s = StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('time')
            ps.print_stats()
            print(s.getvalue())
        t2 = time.time()
        print('**** Done...')
        total_time = t2 - t1
        print(f'Training and validation time: {total_time:5.4f} s')
        if model.perf_counter.num_epochs > 0:
            print(f'Training and validation time per epoch: {total_time / model.perf_counter.num_epochs:5.4f} s')
            print(f'Training and validation throughput: '
                  f'{(model.dataset.train_nsamples * model.perf_counter.num_epochs) / total_time:5.4f} samples/s')
        if model.history_file:
            with open(model.history_file, "w") as f:
                keys = [k for k in history]
                for v in range(len(history[keys[0]])):
                    f.write(' '.join(["%3d" % v] +
                                     [('%20.4f' % history[k][v]) for k in keys]) + '\n')
    # Second (and last) evaluation
    if model.evaluate_on_train:
        if model.rank == 0:
            print('**** Evaluating on test dataset...')
            t1 = time.time()
        _ = model.evaluate_dataset()
        if model.rank == 0:
            t2 = time.time()
            # noinspection PyUnboundLocalVariable
            total_time = t2 - t1
            if not model.evaluate_only:
                print(f'Testing time: {total_time:5.4f} s')
                print(f'Testing throughput: {model.dataset.test_nsamples / total_time:5.4f} samples/s')
    # Print model reports
    if model.rank == 0:
        print_model_reports(model)
    # Barrier and finalize
    if model.comm is not None and model.MPI is not None:
        model.comm.Barrier()
        # The next line is required if running under SLURM (it seems it is not automatically called at exit)
        model.MPI.Finalize()


if __name__ == "__main__":
    main()
