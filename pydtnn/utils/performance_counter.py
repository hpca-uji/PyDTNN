"""Performance monitoring utilities for tracking training and testing metrics."""

import logging
import resource
from collections import defaultdict

import numpy as np

__all__ = ("PerformanceCounter",)

logger = logging.getLogger(__name__)


class PerformanceCounter:
    """Tracks and reports performance metrics including time, throughput, and memory usage."""

    TRAINING, TESTING = range(2)

    def __init__(self):
        """Initializes the performance counter with empty records."""
        self._times_record = defaultdict(lambda: defaultdict(lambda: []))
        self._batch_sizes_record = defaultdict(lambda: defaultdict(lambda: []))
        self._memory_record = defaultdict(lambda: defaultdict(lambda: []))

    #  Public methods and properties

    def add_training_time_and_batch_size(self, epoch, elapsed_time, batch_size):
        """Records training metrics for a specific epoch."""
        self._add_time_and_batch_size(self.TRAINING, epoch, elapsed_time, batch_size)

    def add_testing_time_and_batch_size(self, test_round, elapsed_time, batch_size):
        """Records testing metrics for a specific test round."""
        self._add_time_and_batch_size(self.TESTING, test_round, elapsed_time, batch_size)

    @property
    def training_throughput(self):
        """Returns the overall training throughput in samples per second."""
        return self._throughput(self.TRAINING)

    @property
    def training_throughput_only_last_half_of_each_epoch(self):
        """Returns the training throughput estimated from the last half of each epoch."""
        return self._throughput(self.TRAINING, last_half=True)

    @property
    def num_epochs(self):
        """Returns the total number of training epochs recorded."""
        return len(self._batch_sizes_record[self.TRAINING].keys())

    @property
    def num_evaluations(self):
        """Returns the total number of testing evaluations recorded."""
        return len(self._batch_sizes_record[self.TESTING].keys())

    @property
    def training_time(self):
        """Returns the total training time in seconds."""
        return self._time(self.TRAINING)

    @property
    def training_time_estimated_from_last_half_of_each_epoch(self):
        """Returns the estimated total training time based on the last half of each epoch."""
        return self._time(self.TRAINING, last_half=True)

    @property
    def training_maximum_memory(self):
        """Returns the maximum memory usage recorded during training in KiB."""
        return self._maximum_memory(self.TRAINING)

    @property
    def training_mean_memory(self):
        """Returns the mean memory usage recorded during training in KiB."""
        return self._mean_memory(self.TRAINING)

    @property
    def testing_throughput(self):
        """Returns the overall testing throughput in samples per second."""
        return self._throughput(self.TESTING)

    @property
    def testing_time(self):
        """Returns the total testing time in seconds."""
        return self._time(self.TESTING)

    @property
    def testing_maximum_memory(self):
        """Returns the maximum memory usage recorded during testing in KiB."""
        return self._maximum_memory(self.TESTING)

    @property
    def testing_mean_memory(self):
        """Returns the mean memory usage recorded during testing in KiB."""
        return self._mean_memory(self.TESTING)

    def print_report(self):
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

    def _add_time_and_batch_size(self, where, epoch, elapsed_time, batch_size):
        """Internal helper to record time, batch size, and memory usage."""
        self._times_record[where][epoch].append(elapsed_time)
        self._batch_sizes_record[where][epoch].append(batch_size)
        mem = resource.getrusage(resource.RUSAGE_SELF)[2] + resource.getrusage(resource.RUSAGE_CHILDREN)[2]
        self._memory_record[where][epoch].append(mem)  # KiB in GNU/Linux

    def _time(self, where, last_half=False):
        """Calculates total time for a given phase."""
        return self._sum(self._times_record[where].values(), last_half)

    @staticmethod
    def _sum(arrays, last_half):
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

    def _size(self, where, last_half=False):
        """Calculates total batch size for a given phase."""
        return self._sum(self._batch_sizes_record[where].values(), last_half)

    def _throughput(self, where, last_half=False):
        """Calculates throughput for a given phase."""
        return self._size(where, last_half) / self._time(where, last_half)

    def _maximum_memory(self, where):
        """Calculates maximum memory usage for a given phase."""
        if where == self.TRAINING:
            maximum_memory_per_epoch = [np.max(m_array) for m_array in self._memory_record[where].values()]
            return np.max(maximum_memory_per_epoch)
        else:
            # Consider only the first evaluation
            maximum_memory_first_evaluation = np.max(self._memory_record[where][0])
            return maximum_memory_first_evaluation

    def _mean_memory(self, where):
        """Calculates mean memory usage for a given phase."""
        if where == self.TRAINING:
            mean_memory_per_epoch = [np.mean(m_array) for m_array in self._memory_record[where].values()]
            return np.mean(mean_memory_per_epoch)
        else:
            # Consider only the first evaluation
            mean_memory_first_evaluation = np.mean(self._memory_record[where][0])
            return mean_memory_first_evaluation
