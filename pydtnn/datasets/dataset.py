#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021-22 Universitat Jaume I
#
#  PyDTNN is free software: you can redistribute it and/or modify it under the
#  terms of the GNU General Public License as published by the Free Software
#  Foundation, either version 3 of the License, or (at your option) any later
#  version.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT
#  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
#  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
#  License for more details.
#
#  You should have received a copy of the GNU General Public License along
#  with this program. If not, see <https://www.gnu.org/licenses/>.
#

import queue
import threading
from abc import ABC, abstractmethod

import numpy as np

from pydtnn.utils import PYDTNN_TENSOR_FORMAT_NHWC, PYDTNN_TENSOR_FORMAT_NCHW


class _BackgroundGenerator(threading.Thread):

    def __init__(self, generator, max_prefetch=1):
        super().__init__()
        self.queue = queue.Queue(max_prefetch)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


TRAIN, VAL, TEST = range(3)


class Dataset(ABC):

    def __init__(self, model, train_nsamples, test_nsamples, input_shape, output_shape, max_batches_online=40,
                 force_test_as_validation=False, debug=False):
        self.model = model
        self.max_batches_online = max_batches_online
        self.debug = debug
        self.test_as_validation = self.model.test_as_validation or force_test_as_validation
        self._nsamples = [train_nsamples, 0, test_nsamples]
        # Compute self._nsamples[VAL]
        if self.test_as_validation:
            self._nsamples[VAL] = self._nsamples[TEST]
        else:
            self._nsamples[VAL] = min(self._nsamples[TRAIN] - self.model.nprocs,
                                      max(self.model.nprocs, int(self._nsamples[TRAIN] * self.model.validation_split)))
            self._nsamples[TRAIN] -= self._nsamples[VAL]
        self.input_shape = list(input_shape)
        self.output_shape = list(output_shape)
        self._initial_nsamples = [self._nsamples[TRAIN], self._nsamples[VAL], self._nsamples[TEST]]
        # Offset (in number of samples) and number of samples for the current job for each dataset part
        self._local_offset = [0] * 3
        self._local_nsamples = [0] * 3
        self._local_remaining_nsamples = [-1] * 3  # -1 is used to mark each part as not initialized
        for part in TRAIN, VAL, TEST:
            (self._local_offset[part],
             self._local_nsamples[part],
             self._nsamples[part]
             ) = self._compute_local_workload(self._nsamples[part])
        # Declare _x and _y for train, val and test dataset parts
        self._x = [np.zeros((0, *self.input_shape), dtype=self.model.dtype)] * 3
        self._y = [np.zeros((0, *self.output_shape), dtype=self.model.dtype)] * 3
        if self.model.use_synthetic_data:
            self._data_generator = self._synthetic_data_generator
            self._init_synthetic_data()
        else:
            self._data_generator = self._actual_data_generator
            self._init_actual_data()
        if self.debug:
            self._print_report()

    @property
    def train_nsamples(self):
        return self._nsamples[TRAIN]

    @property
    def val_nsamples(self):
        return self._nsamples[VAL]

    @property
    def test_nsamples(self):
        return self._nsamples[TEST]

    def get_train_val_generator(self):
        return (self._batch_generator(TRAIN),
                self._batch_generator(VAL))

    def get_test_generator(self):
        return self._batch_generator(TEST)

    def _print_report(self):
        if self.model.rank == 0:
            print(f"Initial nsamples:"
                  f" train: {self._initial_nsamples[TRAIN]} "
                  f" val: {self._initial_nsamples[VAL]} "
                  f" test: {self._initial_nsamples[TEST]} "
                  )
        desc = ["train", "val", "test"]
        for part in (TRAIN, VAL, TEST):
            prefix = f"{self.model.rank}: " if part == TRAIN else "   "
            print(f"{prefix}"
                  f" {desc[part]} offset: {self._local_offset[part]}"
                  f" {desc[part]} local nsamples: {self._local_nsamples[part]}"
                  f" {desc[part]} nsamples: {self._nsamples[part]}"
                  )

    def _compute_local_workload(self, nsamples):
        """Computes the offset (in number of samples) and the number of samples for the current rank"""
        new_nsamples = nsamples
        global_batch_size = self.model.batch_size * self.model.nprocs
        batches_per_worker = nsamples // global_batch_size
        remaining_samples = nsamples % global_batch_size
        if not self.model.use_synthetic_data:
            # Version 1) All the data is distributed (which could lead to an unequal distribution)
            # # Instead of assigning all the non-divisible part of the remaining samples to the last process,
            # # it is distributed among all the other workers (i.e, the other workers will have a bit of
            # # work more than the last one). Thus, the use of ceil instead of the integer division.
            # last_batch_nsamples_per_worker = math.ceil(remaining_samples / self.model.nprocs)
            # last_batch_nsamples_last_worker = remaining_samples \
            #                                   - last_batch_nsamples_per_worker * (self.model.nprocs - 1)
            # Version 2) The very last part of the input data is trimmed to ensure an equal distribution
            last_batch_nsamples_per_worker = remaining_samples // self.model.nprocs
            last_batch_nsamples_last_worker = last_batch_nsamples_per_worker
            new_nsamples -= remaining_samples % self.model.nprocs
        else:
            last_batch_nsamples_per_worker = 0
            last_batch_nsamples_last_worker = 0
            new_nsamples = batches_per_worker * self.model.batch_size * self.model.nprocs
        if batches_per_worker > self.model.steps_per_epoch > 0:
            batches_per_worker = self.model.steps_per_epoch
            last_batch_nsamples_per_worker = 0
            last_batch_nsamples_last_worker = 0
            new_nsamples = batches_per_worker * self.model.batch_size * self.model.nprocs
        nsamples_per_worker = batches_per_worker * self.model.batch_size + last_batch_nsamples_per_worker
        local_offset = nsamples_per_worker * self.model.rank
        local_nsamples = \
            nsamples_per_worker \
                if self.model.rank < (self.model.nprocs - 1) \
                else batches_per_worker * self.model.batch_size + last_batch_nsamples_last_worker
        return local_offset, local_nsamples, new_nsamples

    def _init_synthetic_data(self):
        for part in TRAIN, VAL, TEST:
            local_batches = self._local_nsamples[part] // self.model.batch_size
            nsamples = min(local_batches, self.max_batches_online) * self.model.batch_size
            x_shape = [nsamples] + self.input_shape
            if self.model.tensor_format == PYDTNN_TENSOR_FORMAT_NHWC:
                x_shape = [x_shape[i] for i in (0, 2, 3, 1)]
            y_shape = [nsamples] + self.output_shape
            self._x[part] = np.zeros(x_shape, dtype=self.model.dtype, order="C")
            self._y[part] = np.zeros(y_shape, dtype=self.model.dtype, order="C")

    @abstractmethod
    def _init_actual_data(self):
        """Generates initial self._x[] and self._y[]. To be implemented in derived classes."""
        pass

    @staticmethod
    def _nchw2nhwc(x):
        return x.transpose(0, 2, 3, 1).copy()

    @staticmethod
    def _decode_class(y, classes_list):
        """Sets to 1 the corresponding entry in the 2D y array as indicated by the 1D array of classes"""
        y[np.arange(y.shape[0]), classes_list] = 1

    def _synthetic_data_generator(self, part):
        """
        Generates synthetic data for each dataset part returning (slices of) _x[part] and _y[part] initialized in
        _init_synthetic_data().

        The _local_remaining_nsamples[part] vector is used to keep track of:
        - whether a fresh round of the given part should start (if it is -1), or
        - the remaining number of samples for the given part to be yielded.

        Although the data generator should be called in turns: one round of a part until it finishes, then another
        round of the same or a different part, the current implementation, using -1 to mark the end of a round,
        should also support being called for different parts in an interleaved manner. If another version of this
        method is implemented, at least it should raise and exception if a new round begins when a round for another
        part is still in progress.
        """
        for p in (TRAIN, VAL, TEST):
            if self._local_remaining_nsamples[p] == -1:  # If not initialized
                self._local_remaining_nsamples[p] = self._local_nsamples[p]
        while self._local_remaining_nsamples[part] > 0:
            # print()
            # print(f"[part: {part} rank: {self.model.rank}] "
            #       f"{self._local_remaining_nsamples[part]}/{self._x[part].shape[0]}\n")
            if self._local_remaining_nsamples[part] > self._x[part].shape[0]:
                self._local_remaining_nsamples[part] -= self._x[part].shape[0]
                yield self._x[part], self._y[part]
            else:
                remaining_samples = self._local_remaining_nsamples[part]
                self._local_remaining_nsamples[part] = 0
                yield self._x[part][:remaining_samples, ...], self._y[part][:remaining_samples, ...]
        # Mark that a round for part has finished (_local_remaining_nsamples[part] is set to -1 and nothing is yield)
        self._local_remaining_nsamples[part] = -1

    @staticmethod
    def _offset2files(filenames, images_per_file, local_offset, local_nsamples):
        i = local_offset // images_per_file
        offset_in_file = local_offset - i * images_per_file
        output = []
        while local_nsamples:
            nsamples = min(images_per_file - offset_in_file, local_nsamples)
            output.append((filenames[i], offset_in_file, nsamples))
            offset_in_file = 0
            local_nsamples -= nsamples
        return output

    def _actual_data_generator(self, part):
        yield self._x[part], self._y[part]

    def _batch_generator(self, part):
        local_batch_size = self.model.batch_size
        global_batch_size = self.model.batch_size * self.model.nprocs
        generator = self._data_generator(part)
        for x_data, y_data in _BackgroundGenerator(generator):
            local_nsamples = x_data.shape[0]
            s = memoryview(np.arange(local_nsamples))
            if part == TRAIN:
                np.random.shuffle(s)
                if not self.model.use_synthetic_data and (self.model.flip_images or self.model.crop_images):
                    x_data = self._do_data_augmentation(x_data)
            # Initialize end to 0 (in case there are no batches of local_batch_size)
            end = 0
            # Generate batches of size local_batch_size
            for batch_num in range(local_nsamples // local_batch_size):
                start = batch_num * local_batch_size
                end = start + local_batch_size
                indices = s[start:end]
                x_local_batch = x_data[indices, ...]
                y_local_batch = y_data[indices, ...]
                yield x_local_batch, y_local_batch, global_batch_size
            # Generate the last batch (with size < local_batch_size)
            last_batch_size = local_nsamples % local_batch_size
            if last_batch_size > 0:
                start = end
                end = local_nsamples  # = start + last_batch_size
                indices = s[start:end]
                x_local_batch = x_data[indices, ...]
                y_local_batch = y_data[indices, ...]
                yield x_local_batch, y_local_batch, last_batch_size * self.model.nprocs

    def _do_flip_images(self, data):
        if self.model.tensor_format == PYDTNN_TENSOR_FORMAT_NCHW:
            n, c, h, w = data.shape
            width_dim = -1
        else:
            n, h, w, c = data.shape
            width_dim = 2
        limit = min(n, int(n * self.model.flip_images_prob))
        s = np.arange(n)
        np.random.shuffle(s)
        s = s[:limit]
        data[s, ...] = np.flip(data[s, ...], axis=width_dim)
        return data

    def _do_crop_images(self, data):
        if self.model.tensor_format == PYDTNN_TENSOR_FORMAT_NCHW:
            n, c, h, w = data.shape
        else:
            n, h, w, c = data.shape
        crop_size = min(self.model.crop_images_size, h, w)
        limit = min(n, int(n * self.model.crop_images_prob))
        s = np.arange(n)
        np.random.shuffle(s)
        s = s[:limit]
        t = np.random.randint(0, h - crop_size, (limit,))
        ll = np.random.randint(0, w - crop_size, (limit,))
        for i, ri in enumerate(s):
            b, r = t[i] + crop_size, ll[i] + crop_size
            # batch[ri,...] = resize(batch[ri,:,t[i]:b,l[i]:r], (ri.size,c,h,w))
            if self.model.tensor_format == PYDTNN_TENSOR_FORMAT_NCHW:
                data[ri, :, :t[i], :ll[i]] = 0.0
                data[ri, :, b:, r:] = 0.0
            else:
                data[ri, :t[i], :ll[i], :] = 0.0
                data[ri, b:, r:, :] = 0.0
            data[ri, ...] = np.roll(data[ri, ...], np.random.randint(-t[i], (h - b)), axis=1)
            data[ri, ...] = np.roll(data[ri, ...], np.random.randint(-ll[i], (w - r)), axis=2)
        return data

    def _do_data_augmentation(self, x_data):
        # Preserve the original version when producing new data
        x_data = x_data.copy()
        if self.model.flip_images:
            x_data = self._do_flip_images(x_data)
        if self.model.crop_images:
            x_data = self._do_crop_images(x_data)
        return x_data
