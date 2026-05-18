"""
Module for defining the base Scheduler interface and component selection utilities.
"""

import logging

from pydtnn.abstract.base import Base

__all__ = (
    "Scheduler",
)

logger = logging.getLogger(__name__)


class Scheduler(Base):
    """
    Base class for all training schedulers in PyDTNN.
    """

    def __init__(self, verbose: bool):
        """
        Initialize the scheduler.

        Args:
            verbose: Whether to enable logging output.
        """
        super().__init__()
        self.verbose = verbose
        self.epoch_count = 0
        # NOTE: Only used in early_stopping and stop_at_loss.
        # NOTE (cont.): Since there are only 2 classes that uses this variable,
        #   I think it's not necessary to create an abstract class only to store this variable.
        self.stop_training: bool = False

    def __str__(self):
        """
        Return the string representation of the scheduler.
        """
        return f"Scheduler {type(self).__name__}"

    def on_batch_begin(self, *args):
        """
        Hook called at the beginning of a training batch.
        """
        pass

    def on_batch_end(self, *args):
        """
        Hook called at the end of a training batch.
        """
        pass

    def on_epoch_begin(self, *args):
        """
        Hook called at the beginning of a training epoch.
        """
        pass

    def on_epoch_end(self, *args):
        """
        Hook called at the end of a training epoch.
        """
        pass

    def log(self, text: str):
        """
        Log a message if verbose mode is enabled and the process is the primary rank.

        Args:
            text: The message to log.
        """
        if self.verbose and self.model.comm_rank == 0:
            logger.info(f"{self}: {text}")


