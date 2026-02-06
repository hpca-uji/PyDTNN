#!/usr/bin/env python

"""
PyDTNN Benchmark script
"""

import cProfile
import os
from pathlib import Path
import sys
import time
from datetime import datetime

from pydtnn.model import Model
from pydtnn.utils import random
from pydtnn.parser import PydtnnArgumentParser
from pydtnn.utils.best_of import BestOf

ompi_stdout_rank = os.environ.get("OMPI_STDOUT_RANK", None)
if ompi_stdout_rank and os.environ.get("OMPI_COMM_WORLD_RANK", "0") != ompi_stdout_rank:
    sys.stdout = sys.stderr = open(os.devnull, "w")

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


def print_model_reports(model):
    # Print performance counter report
    model.perf_counter.print_report()
    # Print BestOf report
    if model.enable_best_of:
        print()
        BestOf.print_report()


def main():
    # Parse options
    parser = PydtnnArgumentParser()
    config = parser.parse_args()
    # Initialize random seed
    random.seed(config.random_seed)
    # Create model
    model = Model(**vars(config))
    # Print model
    if model.comm_rank == 0:
        model.show_model()
        print()
        model.show_layers()
        print()
    # Print parameters
    if model.comm_rank == 0:
        print('**** Parameters:')
        parser.print_args()
    # First (or unique) evaluation
    if model.evaluate_on_train or model.evaluate_only:
        if model.comm_rank == 0:
            print('**** Evaluating on test dataset...')
            t1 = time.time()
        _ = model.evaluate_dataset()
        if model.comm_rank == 0:
            t2 = time.time()
            # noinspection PyUnboundLocalVariable
            total_time = t2 - t1
            if model.evaluate_only:
                print(f'Testing time: {total_time:5.4f} s')
                print(f'Testing throughput: {model.dataset.test_nsamples / total_time:5.4f} samples/s')
        if model.evaluate_only:
            print_model_reports(model)
            raise SystemExit(0)
    # Barrier
    if model.parallel in ["data"]:
        model.comm.Barrier()
    # Training
    if model.comm_rank == 0:
        # print('**** Model time: ', model.calculate_time())
        print('**** Training...')
        t1 = time.time()
        if model.profile:
            pr = cProfile.Profile()
            pr.enable()
    # Training a model directly from a dataset
    # or alternatively, define any custom data
    # mode.dataset = CustomDataset(model, x, y)
    history = model.train_dataset()
    # Barrier
    if model.parallel == "data":
        model.comm.Barrier()
    # Print performance results and evaluation history
    if model.comm_rank == 0:
        if model.profile:
            pr.disable()
            stamp = datetime.now().isoformat(timespec="seconds").replace(" ", "-").replace(":", "-").replace(".", "-")
            stats = Path(f"profile-{stamp}.stat").resolve()
            pr.dump_stats(stats)
            print(f'Dumped profile stats to: {stats}')
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
                epochs = max(len(v) for v in history.values())
                for epoch in range(epochs):
                    f.write(f"epoch: {epoch}\n")
                    for key in history:
                        f.write(f"    {key}: {history[key][epoch]}\n")
    # Second (and last) evaluation
    if model.evaluate_on_train:
        if model.comm_rank == 0:
            print('**** Evaluating on test dataset...')
            t1 = time.time()
        _ = model.evaluate_dataset()
        if model.comm_rank == 0:
            t2 = time.time()
            # noinspection PyUnboundLocalVariable
            total_time = t2 - t1
            if not model.evaluate_only:
                print(f'Testing time: {total_time:5.4f} s')
                print(f'Testing throughput: {model.dataset.test_nsamples / total_time:5.4f} samples/s')
    # Print model reports
    if model.comm_rank == 0:
        print_model_reports(model)
    # Barrier and finalize
    if model.comm is not None and model.MPI is not None:
        model.comm.Barrier()
        # The next line is required if running under SLURM (it seems it is not automatically called at exit)
        model.MPI.Finalize()


if __name__ == "__main__":
    main()
