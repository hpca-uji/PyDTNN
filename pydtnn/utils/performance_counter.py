"""Performance monitoring utilities for tracking training and testing metrics."""

import logging
import resource
from collections import defaultdict

import numpy as np

from pydtnn.datasets.dataset import Dataset

__all__ = ("PerformanceCounter",)

logger = logging.getLogger(__name__)


class PerformanceCounter:
    """Tracks and reports performance metrics including time, throughput, and memory usage."""

    # Originally: TRAINING, TESTING = range(2); Changed to Dataset.Part

    def __init__(self):
        """Initializes the performance counter with empty records."""
        self._times_record = defaultdict(lambda: defaultdict(lambda: []))
        self._batch_sizes_record = defaultdict(lambda: defaultdict(lambda: []))
        self._memory_record = defaultdict(lambda: defaultdict(lambda: []))

    #  Public methods and properties

    def add_training_time_and_batch_size(self, epoch: int, elapsed_time: float, batch_size: int):
        """Records training metrics for a specific epoch."""
        self._add_time_and_batch_size(Dataset.Part.TRAIN, epoch, elapsed_time, batch_size)

    def add_testing_time_and_batch_size(self, test_round: int, elapsed_time: float, batch_size: int):
        """Records testing metrics for a specific test round."""
        self._add_time_and_batch_size(Dataset.Part.TEST, test_round, elapsed_time, batch_size)

    @property
    def training_throughput(self) -> float:
        """Returns the overall training throughput in samples per second."""
        return self._throughput(Dataset.Part.TRAIN)

    @property
    def training_throughput_only_last_half_of_each_epoch(self) -> float:
        """Returns the training throughput estimated from the last half of each epoch."""
        return self._throughput(Dataset.Part.TRAIN, last_half=True)

    @property
    def num_epochs(self) -> int:
        """Returns the total number of training epochs recorded."""
        return len(self._batch_sizes_record[Dataset.Part.TRAIN].keys())

    @property
    def num_evaluations(self) -> int:
        """Returns the total number of testing evaluations recorded."""
        return len(self._batch_sizes_record[Dataset.Part.TEST].keys())

    @property
    def training_time(self) -> float:
        """Returns the total training time in seconds."""
        return self._time(Dataset.Part.TRAIN)

    @property
    def training_time_estimated_from_last_half_of_each_epoch(self) -> float:
        """Returns the estimated total training time based on the last half of each epoch."""
        return self._time(Dataset.Part.TRAIN, last_half=True)

    @property
    def training_maximum_memory(self) -> int:
        """Returns the maximum memory usage recorded during training in KiB."""
        return self._maximum_memory(Dataset.Part.TRAIN)

    @property
    def training_mean_memory(self) -> float:
        """Returns the mean memory usage recorded during training in KiB."""
        return self._mean_memory(Dataset.Part.TRAIN)

    @property
    def testing_throughput(self) -> float:
        """Returns the overall testing throughput in samples per second."""
        return self._throughput(Dataset.Part.TEST)

    @property
    def testing_time(self):
        """Returns the total testing time in seconds."""
        return self._time(Dataset.Part.TEST)

    @property
    def testing_maximum_memory(self) -> int:
        """Returns the maximum memory usage recorded during testing in KiB."""
        return self._maximum_memory(Dataset.Part.TEST)

    @property
    def testing_mean_memory(self) -> float:
        """Returns the mean memory usage recorded during testing in KiB."""
        return self._mean_memory(Dataset.Part.TEST)

    def print_report(self) -> None:
        """Logs a formatted performance report to the logger."""
        _report = [""]

        if self.num_epochs > 0:
            _report.append(" -------------------------------------")
            _report.append("| Performance counter training report |")
            _report.append(" -------------------------------------")
            _report.append(f"Training time (from model): {self.training_time:5.4f} s")
            _report.append(f"Training time per epoch (from model): {self.training_time / self.num_epochs:5.4f} s")
            _report.append(f"Training throughput (from model): {self.training_throughput:5.4f} samples/s")
            _report.append(f"Training time (from model, estimated from last half of each epoch): {self.training_time_estimated_from_last_half_of_each_epoch:5.4f} s")
            _report.append(f"Training throughput (from model, from last half of each epoch): {self.training_throughput_only_last_half_of_each_epoch:5.4f} samples/s")
            _report.append(f"Training maximum memory allocated: {self.training_maximum_memory / 1024:.2f} MiB")
            _report.append(f"Training mean memory allocated: {self.training_mean_memory / 1024:.2f} MiB")

        if self.num_evaluations > 0:
            _report.append(" ------------------------------------")
            _report.append("| Performance counter testing report |")
            _report.append(" ------------------------------------")
            _report.append(f"Testing time (from model): {self.testing_time / self.num_evaluations:5.4f} s")
            _report.append(f"Testing throughput (from model): {self.testing_throughput:5.4f} samples/s")
            _report.append(f"Testing maximum memory allocated: {self.testing_maximum_memory / 1024:.2f} MiB")
            _report.append(f"Testing mean memory allocated: {self.testing_mean_memory / 1024:.2f} MiB")

        report = "\n".join(_report)
        logger.info(report)

    #  Private methods
    def _add_time_and_batch_size(self, where: Dataset.Part, epoch: int, elapsed_time: float, batch_size: int) -> None:
        """Internal helper to record time, batch size, and memory usage."""
        self._times_record[where][epoch].append(elapsed_time)
        self._batch_sizes_record[where][epoch].append(batch_size)
        # TODO: Check why [2] and move it to a constant or add a comment explain why [2]
        mem = resource.getrusage(resource.RUSAGE_SELF)[2] + resource.getrusage(resource.RUSAGE_CHILDREN)[2]
        self._memory_record[where][epoch].append(mem)  # KiB in GNU/Linux

    def _time(self, where: Dataset.Part, last_half=False) -> float:
        """Calculates total time for a given phase."""
        return self._sum(self._times_record[where].values(), last_half)

    @staticmethod
    def _sum(arrays, last_half: bool) -> int | float:
        # TODO: Add right typing to arrays and check the output type
        """Sums values across records, optionally estimating from the last half of data."""
        # When last_half is True, the total size is estimated from the last half steps of each epoch size
        if not last_half:
            records_per_epoch = [np.sum(array) for array in arrays]
        else:
            records_per_epoch = []
            for array in arrays:
                array_last_half = array[len(array) // 2:]
                if len(array_last_half) > 0:
                    records_per_epoch.append(np.sum(array_last_half) * len(array) / len(array_last_half))
        return np.sum(records_per_epoch)

    def _size(self, where: Dataset.Part, last_half=False):
        # TODO: Add right output's typing
        """Calculates total batch size for a given phase."""
        return self._sum(self._batch_sizes_record[where].values(), last_half)

    def _throughput(self, where: Dataset.Part, last_half=False):
        # TODO: Add right output's typing
        """Calculates throughput for a given phase."""
        return self._size(where, last_half) / self._time(where, last_half)

    def _maximum_memory(self, where: Dataset.Part) -> int:
        """Calculates maximum memory usage for a given phase."""
        match where:
            case Dataset.Part.TRAIN:
                maximum_memory_per_epoch = [np.max(m_array) for m_array in self._memory_record[where].values()]
                return np.max(maximum_memory_per_epoch)
            case Dataset.Part.TEST:
                # Consider only the first evaluation
                maximum_memory_first_evaluation = np.max(self._memory_record[where][0])
                return maximum_memory_first_evaluation
            case _:
                NotImplementedError(f"_maximum_memory not implemented for \"{where}\" case.")
                return 0

    def _mean_memory(self, where: Dataset.Part) -> float:
        """Calculates mean memory usage for a given phase."""
        match where:
            case Dataset.Part.TRAIN:
                mean_memory_per_epoch = [np.mean(m_array) for m_array in self._memory_record[where].values()]
                return np.mean(mean_memory_per_epoch).item()
            case Dataset.Part.TEST:
                # Consider only the first evaluation
                mean_memory_first_evaluation = np.mean(self._memory_record[where][0])
                return mean_memory_first_evaluation.item()
            case _:
                NotImplementedError(f"_maximum_memory not implemented for \"{where}\" case.")
                return 0
