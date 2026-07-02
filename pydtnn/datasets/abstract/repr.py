"""
Dataset module for PyDTNN.

Provides the base Dataset class and utility functions for managing,
transforming, and generating data batches for machine learning models.
"""

import logging

from pydtnn.abstract.base import Base as Baser
from pydtnn.datasets.abstract.base import Base

logger = logging.getLogger(__name__)


class Repr(Base, Baser):
    """
    Abstract base class for all datasets in PyDTNN.

    Defines the interface and common utilities for data partitioning,
    shape management, and format conversion.
    """

    def __init__(self) -> None:
        super().__init__()
        if self.debug:
            self._print_report()

    def _show_props(self) -> dict:
        """Returns a dictionary containing the dataset properties for inspection."""
        props = super()._show_props()

        props["train"] = (self.train_nsamples, *self.input_shape)
        props["val"] = (self.val_nsamples, *self.input_shape)
        props["test"] = (self.test_nsamples, *self.output_shape)

        return props

    def _print_report(self) -> None:
        """Print a summary report of the dataset configuration."""
        report = list[str]()
        if self.model.comm_rank == 0:
            report.append("Initial nsamples:")
            report.append(f" train: {self._initial_nsamples[Base.Part.TRAIN]} ")
            report.append(f" val: {self._initial_nsamples[Base.Part.VAL]} ")
            report.append(f" test: {self._initial_nsamples[Base.Part.TEST]} ")

        desc = ["train", "val", "test"]
        for part in (Base.Part.TRAIN, Base.Part.VAL, Base.Part.TEST):
            prefix = f"{self.model.rank}: " if part is Base.Part.TRAIN else "   "
            report.append(f"{prefix}")
            report.append(f" {desc[part]} offset: {self._local_offset[part]}")
            report.append(f" {desc[part]} local nsamples: {self._local_nsamples[part]}")
            report.append(f" {desc[part]} nsamples: {self._nsamples[part]}")

        logger.info("\n".join(report))
