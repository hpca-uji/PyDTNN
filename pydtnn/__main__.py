#!/usr/bin/env python
"""PyDTNN Benchmark script."""

import logging
import logging.config
import os
import platform
import sys
import time
from argparse import Namespace
from importlib import metadata, resources

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
    from pydtnn import environ as devices
    from pydtnn import package_name, rank, timestamp
    from pydtnn.utils import header, rand

    # Initialize
    rand.seed(config.random_seed)
    if rank == 0:
        header("PyDTNN benchmark")

    # Environment
    packages = {platform.python_implementation().lower(): platform.python_version()}
    for packs in metadata.packages_distributions().values():
        for pack in packs:
            packages[pack] = metadata.version(pack)
    environ = {"timestamp": timestamp, "packages": packages, "devices": devices}

    # Create model
    from pydtnn.model import Model

    model = Model(**vars(config))
    model.history_append(environ)
    model._ensure_runnable()

    # Print model
    if model.comm_rank == 0:
        logger.info(str(config))
        model.show_model()
        model.show_layers()

    # First (or unique) evaluation
    if model.evaluate_on_train or model.evaluate_only:
        if model.comm_rank == 0:
            header("Testing...")
        tic = time.time()
        _ = model.evaluate()
        toc = time.time()
        delta = toc - tic
        if model.comm_rank == 0:
            if model.evaluate_only:
                logger.info(f"Testing time: {delta:5.4f} s")
                logger.info(
                    f"Testing throughput: {model.dataset.test_nsamples / delta:5.4f} samples/s"
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
    # Training a model directly from a dataset
    # or alternatively, define any custom data
    # mode.dataset = CustomDataset(model, x, y)
    tic = time.time()
    model.train()
    toc = time.time()
    delta = toc - tic

    # Barrier
    if model.comm:
        model.comm.Barrier()

    # Print performance results and evaluation history
    if model.comm_rank == 0:
        logger.info(f"Training and validation time: {delta:5.4f} s")
        if model.perf_counter.num_epochs > 0:
            logger.info(
                "Training and validation time per epoch:"
                f" {delta / model.perf_counter.num_epochs:5.4f} s"
            )
            logger.info(
                "Training and validation throughput:"
                f" {(model.dataset.train_nsamples * model.perf_counter.num_epochs) / delta:5.4f} samples/s"
            )

    # Second (and last) evaluation
    if model.evaluate_on_train:
        if model.comm_rank == 0:
            header("Testing...")
        tic = time.time()
        model.evaluate()
        toc = time.time()
        delta = toc - tic

        if model.comm_rank == 0:
            if not model.evaluate_only:
                logger.info(f"Testing time: {delta:5.4f} s")
                logger.info(
                    f"Testing throughput: {model.dataset.test_nsamples / delta:5.4f} samples/s"
                )

    # Print model reports
    if model.comm_rank == 0:
        if model.profile:
            model.profiler.dump_stats(f"{package_name}-{timestamp}.prof")
        model.history_append(model.perf_counter._show_props())
        model.perf_counter.print_report()

    # Barrier and finalize
    if model.comm and model.MPI:
        model.comm.Barrier()
        # The next line is required if running under SLURM (it seems it is not
        # automatically called at exit)
        model.MPI.Finalize()


if __name__ == "__main__":
    raise SystemExit(_start())
