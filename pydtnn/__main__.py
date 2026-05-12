#!/usr/bin/env python

"""
PyDTNN Benchmark script.
"""

import cProfile
import logging
import logging.config
import os
import sys
import time
from importlib import resources
from pathlib import Path

import yaml

from pydtnn import timestamp, utils
from pydtnn.utils.debug import traceback_context
from pydtnn.utils.parser import ArgumentParser
from pydtnn.utils.serial import NumpyYaml

__all__ = ("main",)

logger = logging.getLogger(__name__)
log_conf = yaml.safe_load(resources.read_text("pydtnn", "logger.yaml"))
logging.config.dictConfig(log_conf)


ompi_stdout_rank = os.environ.get("OMPI_STDOUT_RANK", None)
if ompi_stdout_rank and os.environ.get("OMPI_COMM_WORLD_RANK", "0") != ompi_stdout_rank:
    sys.stdout = sys.stderr = open(os.devnull, "w")

Extrae_tracing = False
if os.environ.get("EXTRAE_ON", None) == "1":
    TracingLibrary = "libptmpitrace.so"
    import pyextrae.common.extrae as pyextrae  # type: ignore

    pyextrae.startTracing(TracingLibrary)
    Extrae_tracing = True


def _start() -> int:
    """System entry point"""
    parser = ArgumentParser()
    config = parser.parse_args()

    # CLI defaults
    config.model_name = config.model_name or "simplecnn"
    config.dataset_name = config.dataset_name or "mnist"

    with traceback_context():
        return main(config)  # type: ignore


def main(config):
    """Application entry point"""
    # Initialize random seed
    from pydtnn.utils import random

    random.seed(config.random_seed)
    # Create model
    from pydtnn.model import Model

    model = Model(**vars(config))
    model._ensure_model_runnable()
    # Print model
    if model.comm_rank == 0:
        logger.info(str(config))
        model.show_model()
        model.show_layers()
    # First (or unique) evaluation
    if model.evaluate_on_train or model.evaluate_only:
        if model.comm_rank == 0:
            logger.info("**** Evaluating on test dataset...")
            t1 = time.time()
        _ = model.evaluate()
        if model.comm_rank == 0:
            t2 = time.time()
            # noinspection PyUnboundLocalVariable
            total_time = t2 - t1
            if model.evaluate_only:
                logger.info(f"Testing time: {total_time:5.4f} s")
                logger.info(f"Testing throughput: {model.dataset.test_nsamples / total_time:5.4f} samples/s")
        if model.evaluate_only:
            model.perf_counter.print_report()
            raise SystemExit(0)
    # Barrier
    if model.parallel_data:
        assert model.comm
        model.comm.Barrier()
    # Training
    if model.comm_rank == 0:
        # print('**** Model time: ', model.calculate_time())
        logger.info("**** Training...")
        t1 = time.time()
        if model.profile:
            pr = cProfile.Profile()
            pr.enable()
    # Training a model directly from a dataset
    # or alternatively, define any custom data
    # mode.dataset = CustomDataset(model, x, y)
    history = model.train()
    # Barrier
    if model.parallel_data:
        assert model.comm
        model.comm.Barrier()
    # Print performance results and evaluation history
    if model.comm_rank == 0:
        if model.profile:
            pr.disable()
            path = Path(f"profile-{timestamp}.stat").resolve()
            pr.dump_stats(path)
            logger.info(f"Dumped profile stats to: {path}")
        t2 = time.time()
        logger.info("**** Done...")
        total_time = t2 - t1
        logger.info(f"Training and validation time: {total_time:5.4f} s")
        if model.perf_counter.num_epochs > 0:
            logger.info(f"Training and validation time per epoch: {total_time / model.perf_counter.num_epochs:5.4f} s")
            logger.info(f"Training and validation throughput: {(model.dataset.train_nsamples * model.perf_counter.num_epochs) / total_time:5.4f} samples/s")
    # Store history information
    if model.history_file:
        history_file = utils.string_substitute(model.history_file, rank=model.comm_rank)
        if history_file != model.history_file or model.comm_rank == 0:
            path = Path(history_file).resolve()
            events = []
            epochs = max(len(v) for v in history.values())
            for epoch in range(epochs):
                events.append({"epoch": epoch} | {key: history[key][epoch] for key in history})
            with open(path, "w") as f:
                yaml.dump_all(events, f, NumpyYaml, allow_unicode=True, sort_keys=False)
            logger.info(f"Dumped metric history to: {path}")
    # Second (and last) evaluation
    if model.evaluate_on_train:
        if model.comm_rank == 0:
            logger.info("**** Evaluating on test dataset...")
            t1 = time.time()
        _ = model.evaluate()
        if model.comm_rank == 0:
            t2 = time.time()
            # noinspection PyUnboundLocalVariable
            total_time = t2 - t1
            if not model.evaluate_only:
                logger.info(f"Testing time: {total_time:5.4f} s")
                logger.info(f"Testing throughput: {model.dataset.test_nsamples / total_time:5.4f} samples/s")
    # Print model reports
    if model.comm_rank == 0:
        model.perf_counter.print_report()
    # Barrier and finalize
    if model.comm is not None and model.MPI is not None:
        model.comm.Barrier()
        # The next line is required if running under SLURM (it seems it is not automatically called at exit)
        model.MPI.Finalize()


if __name__ == "__main__":
    raise SystemExit(_start())
