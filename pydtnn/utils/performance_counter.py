import resource
from collections import defaultdict

import numpy as np


class PerformanceCounter:
    TRAINING, TESTING = range(2)

    def __init__(self):
        self._times_record = defaultdict(lambda: defaultdict(lambda: []))
        self._batch_sizes_record = defaultdict(lambda: defaultdict(lambda: []))
        self._memory_record = defaultdict(lambda: defaultdict(lambda: []))

    # -------------------------------
    #  Public methods and properties
    # -------------------------------

    def add_training_time_and_batch_size(self, epoch, elapsed_time, batch_size):
        self._add_time_and_batch_size(self.TRAINING, epoch, elapsed_time, batch_size)

    def add_testing_time_and_batch_size(self, test_round, elapsed_time, batch_size):
        self._add_time_and_batch_size(self.TESTING, test_round, elapsed_time, batch_size)

    @property
    def training_throughput(self):
        return self._throughput(self.TRAINING)

    @property
    def training_throughput_only_last_half_of_each_epoch(self):
        return self._throughput(self.TRAINING, last_half=True)

    @property
    def num_epochs(self):
        return len(self._batch_sizes_record[self.TRAINING].keys())

    @property
    def num_evaluations(self):
        return len(self._batch_sizes_record[self.TESTING].keys())

    @property
    def training_time(self):
        return self._time(self.TRAINING)

    @property
    def training_time_estimated_from_last_half_of_each_epoch(self):
        return self._time(self.TRAINING, last_half=True)

    @property
    def training_maximum_memory(self):
        return self._maximum_memory(self.TRAINING)

    @property
    def training_mean_memory(self):
        return self._mean_memory(self.TRAINING)

    @property
    def testing_throughput(self):
        return self._throughput(self.TESTING)

    @property
    def testing_time(self):
        return self._time(self.TESTING)

    @property
    def testing_maximum_memory(self):
        return self._maximum_memory(self.TESTING)

    @property
    def testing_mean_memory(self):
        return self._mean_memory(self.TESTING)

    def print_report(self):
        if self.num_epochs > 0:
            print(" -------------------------------------")
            print("| Performance counter training report |")
            print(" -------------------------------------")
            print(f'Training time (from model): {self.training_time:5.4f} s')
            print(f'Training time per epoch (from model): '
                  f'{self.training_time / self.num_epochs:5.4f} s')
            print(f'Training throughput (from model): {self.training_throughput:5.4f} samples/s')
            print(f'Training time (from model, estimated from last half of each epoch): '
                  f'{self.training_time_estimated_from_last_half_of_each_epoch:5.4f} s')
            print(f'Training throughput (from model, from last half of each epoch): '
                  f'{self.training_throughput_only_last_half_of_each_epoch:5.4f} samples/s')
            print(f'Training maximum memory allocated: '
                  f'{self.training_maximum_memory / 1024:.2f} MiB')
            print(f'Training mean memory allocated: '
                  f'{self.training_mean_memory / 1024:.2f} MiB')

        if self.num_evaluations > 0:
            print(" ------------------------------------")
            print("| Performance counter testing report |")
            print(" ------------------------------------")
            print(f'Testing time (from model): {self.testing_time / self.num_evaluations:5.4f} s')
            print(f'Testing throughput (from model): {self.testing_throughput:5.4f} samples/s')
            print(f'Testing maximum memory allocated: ',
                  f'{self.testing_maximum_memory / 1024:.2f} MiB')
            print(f'Testing mean memory allocated: ',
                  f'{self.testing_mean_memory / 1024:.2f} MiB')

    # -------------------------------
    #  Private methods
    # -------------------------------

    def _add_time_and_batch_size(self, where, epoch, elapsed_time, batch_size):
        self._times_record[where][epoch].append(elapsed_time)
        self._batch_sizes_record[where][epoch].append(batch_size)
        mem = (resource.getrusage(resource.RUSAGE_SELF)[2]
               + resource.getrusage(resource.RUSAGE_CHILDREN)[2])
        self._memory_record[where][epoch].append(mem)  # KiB in GNU/Linux

    def _time(self, where, last_half=False):
        return self._sum(self._times_record[where].values(), last_half)

    @staticmethod
    def _sum(arrays, last_half):
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
        return self._sum(self._batch_sizes_record[where].values(), last_half)

    def _throughput(self, where, last_half=False):
        return self._size(where, last_half) / self._time(where, last_half)

    def _maximum_memory(self, where):
        if where == self.TRAINING:
            maximum_memory_per_epoch = [np.max(m_array) for m_array in self._memory_record[where].values()]
            return np.max(maximum_memory_per_epoch)
        else:
            # Consider only the first evaluation
            maximum_memory_first_evaluation = np.max(self._memory_record[where][0])
            return maximum_memory_first_evaluation

    def _mean_memory(self, where):
        if where == self.TRAINING:
            mean_memory_per_epoch = [np.mean(m_array)
                                     for m_array in self._memory_record[where].values()]
            return np.mean(mean_memory_per_epoch)
        else:
            # Consider only the first evaluation
            mean_memory_first_evaluation = np.mean(self._memory_record[where][0])
            return mean_memory_first_evaluation
