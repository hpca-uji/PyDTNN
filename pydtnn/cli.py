#!/usr/bin/env python

"""
PyDTNN Benchmark script
"""

import numpy as np
import yaml
from contextlib import contextmanager, nullcontext
from traceback import TracebackException
from datetime import datetime
from pathlib import Path
from importlib import resources
import logging.config
import cProfile
import time
import sys
import os
import logging

from pydtnn import utils
logger = logging.getLogger(__name__)


ompi_stdout_rank = os.environ.get("OMPI_STDOUT_RANK", None)
if ompi_stdout_rank and os.environ.get("OMPI_COMM_WORLD_RANK", "0") != ompi_stdout_rank:
    sys.stdout = sys.stderr = open(os.devnull, "w")

Extrae_tracing = False
if os.environ.get("EXTRAE_ON", None) == "1":
    TracingLibrary = "libptmpitrace.so"
    import pyextrae.common.extrae as pyextrae

    pyextrae.startTracing(TracingLibrary)
    Extrae_tracing = True

timestamp = datetime.now().isoformat(timespec="seconds").replace(" ", "-").replace(":", "-").replace(".", "-")


def show_options(params):
    for arg in vars(params):
        if arg != "comm":
            logger.info(f'  {arg:31s}: {str(getattr(params, arg)):s}')
            # print(f'  --{arg:s}={str(getattr(params, arg)):s} \\')


def print_model_reports(model):
    # Print performance counter report
    model.perf_counter.print_report()
    # Print BestOf report
    # if model.enable_best_of:
    #     BestOf.print_report()


class HistoryDumper(yaml.SafeDumper):
    def represent_ndarray(self, data):
        return self.represent_scalar('!ndarray', repr(data), style="|")


HistoryDumper.add_representer(np.ndarray, HistoryDumper.represent_ndarray)


@contextmanager
def traceback_context():
    try:
        yield
    except Exception as exc:
        path = Path(f"traceback-{timestamp}.txt").resolve()
        with Path(path).open(mode="w") as file:
            TracebackException.from_exception(exc, capture_locals=True).print(file=file)
        logger.info(f'Dumped traceback details to: {path}')
        raise


def main():
    log_conf = yaml.safe_load(resources.read_text("pydtnn", "logger.yaml"))
    logging.config.dictConfig(log_conf)

    from pydtnn.model import Model
    from pydtnn.utils import random
    from pydtnn.parser import PydtnnArgumentParser
    # from pydtnn.utils.best_of import BestOf

    # Parse options
    parser = PydtnnArgumentParser()
    config = parser.parse_args()
    exc_tracer = traceback_context if config.traceback else nullcontext
    # Initialize random seed
    random.seed(config.random_seed)
    # Create model
    with exc_tracer():
        model = Model(**vars(config))
    # Print model
    if model.comm_rank == 0:
        model.show_model()
        model.show_layers()
    # Print parameters
    if model.comm_rank == 0:
        logger.info('**** Parameters:')
        parser.print_args()
    # First (or unique) evaluation
    if model.evaluate_on_train or model.evaluate_only:
        if model.comm_rank == 0:
            logger.info('**** Evaluating on test dataset...')
            t1 = time.time()
        with exc_tracer():
            _ = model.evaluate()
        if model.comm_rank == 0:
            t2 = time.time()
            # noinspection PyUnboundLocalVariable
            total_time = t2 - t1
            if model.evaluate_only:
                logger.info(f'Testing time: {total_time:5.4f} s')
                logger.info(f'Testing throughput: {model.dataset.test_nsamples / total_time:5.4f} samples/s')
        if model.evaluate_only:
            print_model_reports(model)
            raise SystemExit(0)
    # Barrier
    if model.parallel_data:
        model.comm.Barrier()
    # Training
    if model.comm_rank == 0:
        # print('**** Model time: ', model.calculate_time())
        logger.info('**** Training...')
        t1 = time.time()
        if model.profile:
            pr = cProfile.Profile()
            pr.enable()
    # Training a model directly from a dataset
    # or alternatively, define any custom data
    # mode.dataset = CustomDataset(model, x, y)
    with exc_tracer():
        history = model.train()
    # Barrier
    if model.parallel_data:
        model.comm.Barrier()
    # Print performance results and evaluation history
    if model.comm_rank == 0:
        if model.profile:
            pr.disable()
            path = Path(f"profile-{timestamp}.stat").resolve()
            pr.dump_stats(path)
            logger.info(f'Dumped profile stats to: {path}')
        t2 = time.time()
        logger.info('**** Done...')
        total_time = t2 - t1
        logger.info(f'Training and validation time: {total_time:5.4f} s')
        if model.perf_counter.num_epochs > 0:
            logger.info(f'Training and validation time per epoch: {total_time / model.perf_counter.num_epochs:5.4f} s')
            logger.info(f'Training and validation throughput: '
                        f'{(model.dataset.train_nsamples * model.perf_counter.num_epochs) / total_time:5.4f} samples/s')
    # Store history information
    history_file = utils.string_substitute(model.history_file, rank=model.comm_rank)
    if history_file != model.history_file or model.comm_rank == 0:
        events = []
        epochs = max(len(v) for v in history.values())
        for epoch in range(epochs):
            events.append({"epoch": epoch} | {key: history[key][epoch] for key in history})
        with open(history_file, "w") as f:
            yaml.dump_all(events, f, HistoryDumper, allow_unicode=True, sort_keys=False)
    # Second (and last) evaluation
    if model.evaluate_on_train:
        if model.comm_rank == 0:
            logger.info('**** Evaluating on test dataset...')
            t1 = time.time()
        with exc_tracer():
            _ = model.evaluate()
        if model.comm_rank == 0:
            t2 = time.time()
            # noinspection PyUnboundLocalVariable
            total_time = t2 - t1
            if not model.evaluate_only:
                logger.info(f'Testing time: {total_time:5.4f} s')
                logger.info(f'Testing throughput: {model.dataset.test_nsamples / total_time:5.4f} samples/s')
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
