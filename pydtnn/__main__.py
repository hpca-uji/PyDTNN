#!/usr/bin/env python
"""PyDTNN Benchmark script."""

import cProfile
import logging
import logging.config
import os
import sys
import time
from argparse import Namespace
from importlib import resources
from pathlib import Path

import yaml

__all__ = ("main",)


logger = logging.getLogger(__name__)


def _start() -> int:
    """System entry point"""
    if os.environ.get("PYEXTRAE"):
        import pyextrae.common.extrae as pyextrae

        pyextrae.startTracing("libptmpitrace.so")

    from pydtnn import rank

    if rank == 0:
        config = yaml.safe_load(resources.read_text("pydtnn", "logger.yaml"))
        logging.config.dictConfig(config)
    else:
        sys.stdout = sys.stderr = open(os.devnull, "w")

    from pydtnn.utils.parser import ArgumentParser

    parser = ArgumentParser()
    config = parser.parse_args()

    from pydtnn.utils.debug import traceback_context

    with traceback_context():
        return main(config) or 0


def main(config: Namespace) -> None:  # noqa: C901
    """Application entry point"""

    # Initialize random seed
    from pydtnn.utils import header, rand

    rand.seed(config.random_seed)

    # Create model
    from pydtnn.model import Model

    pr = cProfile.Profile()
    model = Model(**vars(config))

    if model.comm_rank == 0:
        header("PyDTNN benchmark")

    model._ensure_runnable()

    from pydtnn import timestamp, utils

    # Print model
    if model.comm_rank == 0:
        logger.info(str(config))
        model.show_model()
        model.show_layers()

    # First (or unique) evaluation
    if model.evaluate_on_train or model.evaluate_only:
        if model.comm_rank == 0:
            header("Testing...")
        t1 = time.time()
        _ = model.evaluate()
        t2 = time.time()
        total_time = t2 - t1
        if model.comm_rank == 0:
            if model.evaluate_only:
                logger.info(f"Testing time: {total_time:5.4f} s")
                logger.info(
                    f"Testing throughput: {model.dataset.test_nsamples / total_time:5.4f} samples/s"
                )
        if model.evaluate_only:
            if model.comm_rank == 0:
                model.perf_counter.print_report()
            raise SystemExit(0)

    # Barrier
    if model.comm:
        model.comm.Barrier()

    # Training
    if model.comm_rank == 0:
        # print('# Model time: ', model.calculate_time())
        header("Training...")
        if model.profile:
            pr.enable()
    # Training a model directly from a dataset
    # or alternatively, define any custom data
    # mode.dataset = CustomDataset(model, x, y)
    t1 = time.time()
    model.train()
    t2 = time.time()
    total_time = t2 - t1

    # Barrier
    if model.comm:
        model.comm.Barrier()

    # Print performance results and evaluation history
    if model.comm_rank == 0:
        if model.profile:
            pr.disable()
            path = Path(f"profile-{timestamp}.stat").resolve()
            pr.dump_stats(path)
            logger.info(f"Dumped profile stats to: {path}")
        logger.info(f"Training and validation time: {total_time:5.4f} s")
        if model.perf_counter.num_epochs > 0:
            logger.info(
                "Training and validation time per epoch:"
                f" {total_time / model.perf_counter.num_epochs:5.4f} s"
            )
            logger.info(
                "Training and validation throughput:"
                f" {(model.dataset.train_nsamples * model.perf_counter.num_epochs) / total_time:5.4f} samples/s"
            )

    # Second (and last) evaluation
    if model.evaluate_on_train:
        if model.comm_rank == 0:
            header("Testing...")
            t1 = time.time()
        _ = model.evaluate()
        if model.comm_rank == 0:
            t2 = time.time()
            total_time = t2 - t1
            if not model.evaluate_only:
                logger.info(f"Testing time: {total_time:5.4f} s")
                logger.info(
                    f"Testing throughput: {model.dataset.test_nsamples / total_time:5.4f} samples/s"
                )

    # Store history information
    if model.history_file:
        history_file = utils.string_substitute(model.history_file, rank=model.comm_rank)
        if history_file != model.history_file or model.comm_rank == 0:
            from pydtnn.utils.serial import NumpyYaml

            history = model.history
            path = Path(history_file).resolve()
            with open(path, "w") as f:
                yaml.dump_all(history, f, NumpyYaml, allow_unicode=True, sort_keys=False)
            logger.info(f"Dumped metric history to: {path}")

    # Print model reports
    if model.comm_rank == 0:
        model.perf_counter.print_report()

    # Barrier and finalize
    if model.comm and model.MPI:
        model.comm.Barrier()
        # The next line is required if running under SLURM (it seems it is not
        # automatically called at exit)
        model.MPI.Finalize()


if __name__ == "__main__":
    raise SystemExit(_start())
