#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021-25 Universitat Jaume I
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
import warnings
import itertools
import threading
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image

from pydtnn.utils import PYDTNN_TENSOR_FORMAT, string_substitute
from typing import TYPE_CHECKING, Generator
if TYPE_CHECKING:
    from pydtnn.model import Model
from enum import IntEnum
from pydtnn.backends.gpu import TensorGPU
type Array = np.ndarray | TensorGPU
type shape_t  = tuple[int, ...]

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

    def __next__(self) -> tuple[Array, Array]:
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self

class DatasetEnum(IntEnum):
    TRAIN = 0
    VAL = 1
    TEST = 2
# --- END DatasetEnum --- #


class Dataset(ABC):
    # NOTE: Dataset(input_shape) is expected to be in NCHW format
    # NOTE: Dataset.data_generator(x) is expected to be in model.tensor_format format
    # NOTE: Dataset.data_generator(y) is expected to be in NC format

    def __init__(self, model: "Model", train_nsamples:int, test_nsamples:int, input_shape:shape_t, output_shape:shape_t, 
                 max_batches_online = 40, force_test_as_validation=False, debug=False):

        if len(input_shape) != 3:
            warnings.warn(f"Input shape does not have 3 dimensions ({input_shape}), it may cause issues!", RuntimeWarning)
        # if len(input_shape) == 3 and not (input_shape[0] < input_shape[2]):
        elif not (input_shape[0] < input_shape[2]):
            warnings.warn(f"Dataset input_shape {input_shape} may not be in NCHW format, regardless of model format!", RuntimeWarning)

        # TODO: Check if this makes sense.
        if len(output_shape) != 1:
            warnings.warn(f"Output shape does not have 1 dimension ({output_shape}), it may cause issues!", RuntimeWarning)

        self.model:Model = model
        self.max_batches_online:int = max_batches_online
        self.debug:bool = debug
        self.test_as_validation:bool = self.model.test_as_validation or force_test_as_validation
        self._nsamples:list[int, int, int] = [train_nsamples, 0, test_nsamples]

        # Compute self._nsamples[DatasetEnum.VAL]
        if self.test_as_validation:
            self._nsamples[DatasetEnum.VAL] = self._nsamples[DatasetEnum.TEST]
        else:
            self._nsamples[DatasetEnum.VAL] = min(self._nsamples[DatasetEnum.TRAIN] - self.model.nprocs,
                                                   max(self.model.nprocs, 
                                                       int(self._nsamples[DatasetEnum.TRAIN] * self.model.validation_split)))
            self._nsamples[DatasetEnum.TRAIN] -= self._nsamples[DatasetEnum.VAL]
        
        if self.model.resize:
            self.input_shape = list((input_shape[0], *self.model.resize_dimension))
            self.resize_shape = self.input_shape if self.model.tensor_format is PYDTNN_TENSOR_FORMAT.NCHW else (self.input_shape[1], self.input_shape[2], self.input_shape[0])
            self.real_input_shape = list(input_shape)
        else:
            self.input_shape = list(input_shape)
            self.real_input_shape = self.input_shape
        self.output_shape = list(output_shape)

        self._initial_nsamples = [self._nsamples[DatasetEnum.TRAIN], self._nsamples[DatasetEnum.VAL], self._nsamples[DatasetEnum.TEST]]
        # Offset (in number of samples) and number of samples for the current job for each dataset part
        self._local_offset = [0] * 3
        self._local_nsamples = [0] * 3
        self._local_remaining_nsamples = [-1] * 3  # -1 is used to mark each part as not initialized

        for part in DatasetEnum.TRAIN, DatasetEnum.VAL, DatasetEnum.TEST:
            (self._local_offset[part],
             self._local_nsamples[part],
             self._nsamples[part]
             ) = self._compute_local_workload(self._nsamples[part])
            
        # Declare _x and _y for train, val and test dataset parts
        # FIXME: This input shape must be the real one.
        self._x = [np.zeros((0, *self.real_input_shape), dtype=self.model.dtype)] * len(DatasetEnum)
        self._y = [np.zeros((0, *self.output_shape), dtype=self.model.dtype)] * len(DatasetEnum)

        if self.model.use_synthetic_data:
            self._data_generator = self._synthetic_data_generator
            self._init_synthetic_data()
        else:
            self._data_generator = self._actual_data_generator
            self._init_actual_data()

        if self.debug:
            self._print_report()

    def export(self, split_weights: list[float] | None = None):
        """Export dataset (possibly split and rank specific)"""

        # Get split weights
        if split_weights is None:
            split_weights = list(map(float, self.model.dataset_export_split_weights.split(",")))

        # Data generators
        gen_train = self._data_generator(DatasetEnum.TRAIN)
        gen_val = self._data_generator(DatasetEnum.VAL)
        gen_test = self._data_generator(DatasetEnum.TEST)

        # Reconstruct validation split
        if self.test_as_validation:
            gen_test = itertools.chain(gen_test, gen_val)
        else:
            gen_train = itertools.chain(gen_train, gen_val)

        # Array from generators
        x_train, y_train = map(np.concat, zip(*gen_train))
        x_test, y_test = map(np.concat, zip(*gen_test))

        # Ensure dataset is in NCHW
        match self.model.tensor_format:
            case PYDTNN_TENSOR_FORMAT.NCHW:
                pass
            case PYDTNN_TENSOR_FORMAT.NHWC:
                x_train = x_train.transpose(0, 3, 1, 2)
                x_test = x_test.transpose(0, 3, 1, 2)
            case _:
                raise NotImplementedError(f"Unsupported tensor format {self.model.tensor_format}")

        # Ensure dataset is in float64
        match self.model.dtype:
            case np.float64:
                pass
            case np.float32:
                x_train, y_train = x_train.astype(np.float64), y_train.astype(np.float64)
                x_test, y_test = x_test.astype(np.float64), y_test.astype(np.float64)
            case _:
                raise NotImplementedError(f"Unsupported model dtype {self.model.dtype}")

        # Calculate percentage splits
        total = sum(split_weights)
        split_percentage = [weight / total for weight in itertools.accumulate(split_weights)]

        # Split arrays
        np_splits = np.array(split_percentage[:-1])
        x_train = np.split(x_train, (len(x_train) * np_splits).astype(int))
        y_train = np.split(y_train, (len(y_train) * np_splits).astype(int))
        x_test = np.split(x_test, (len(x_test) * np_splits).astype(int))
        y_test = np.split(y_test, (len(y_test) * np_splits).astype(int))

        # Save arrays
        for split, (x_train, y_train, x_test, y_test) in enumerate(zip(x_train, y_train, x_test, y_test)):
            path = string_substitute(self.model.dataset_raw_path, split=split)

            # Export dataset
            np.savez_compressed(path,
                                x_train=x_train,
                                y_train=y_train,
                                x_test=x_test,
                                y_test=y_test)

            # Debug information
            if self.debug:
                print(f"Export: {path}")
                print(f"x_train: {x_train.shape}")
                print(f"y_train: {y_train.shape}")
                print(f"x_test: {x_test.shape}")
                print(f"y_test: {y_test.shape}")

    @property
    def train_nsamples(self):
        return self._nsamples[DatasetEnum.TRAIN]

    @property
    def val_nsamples(self):
        return self._nsamples[DatasetEnum.VAL]

    @property
    def test_nsamples(self):
        return self._nsamples[DatasetEnum.TEST]

    def get_train_val_generator(self) -> tuple[Generator[tuple[Array, Array, int]], Generator[tuple[Array, Array, int]]]:
        return (self._batch_generator(DatasetEnum.TRAIN),
                self._batch_generator(DatasetEnum.VAL))

    def get_test_generator(self) -> Generator[tuple[Array, Array, int]]:
        return self._batch_generator(DatasetEnum.TEST)

    def _print_report(self):
        if self.model.comm_rank == 0:
            print(f"Initial nsamples:"
                  f" train: {self._initial_nsamples[DatasetEnum.TRAIN]} "
                  f" val: {self._initial_nsamples[DatasetEnum.VAL]} "
                  f" test: {self._initial_nsamples[DatasetEnum.TEST]} "
                  )
        desc = ["train", "val", "test"]
        for part in (DatasetEnum.TRAIN, DatasetEnum.VAL, DatasetEnum.TEST):
            prefix = f"{self.model.rank}: " if part is DatasetEnum.TRAIN else "   "
            print(f"{prefix}"
                  f" {desc[part]} offset: {self._local_offset[part]}"
                  f" {desc[part]} local nsamples: {self._local_nsamples[part]}"
                  f" {desc[part]} nsamples: {self._nsamples[part]}"
                  )

    def _compute_local_workload(self, nsamples):
        """Computes the offset (in number of samples) and the number of samples for the current rank"""

        # Reduce nsamples according to steps per epoch
        global_batch_size = self.model.batch_size * self.model.nprocs
        batches_per_worker = nsamples / global_batch_size
        if batches_per_worker > self.model.steps_per_epoch > 0:
            batches_per_worker = self.model.steps_per_epoch
            nsamples = batches_per_worker * global_batch_size

        # Calculate nsamples per worker
        nsamples_per_worker, big_workers = divmod(nsamples, self.model.nprocs)
        nsamples_per_big_worker = nsamples_per_worker + 1

        # Calculate local values
        if self.model.rank < big_workers:
            local_nsamples = nsamples_per_big_worker
            local_offset = self.model.rank * nsamples_per_big_worker
        else:
            local_nsamples = nsamples_per_worker
            local_offset = nsamples_per_big_worker * big_workers + nsamples_per_worker * (self.model.rank - big_workers)

        return local_offset, local_nsamples, nsamples

    def _init_synthetic_data(self):
        for part in DatasetEnum.TRAIN, DatasetEnum.VAL, DatasetEnum.TEST:
            local_batches = self._local_nsamples[part] // self.model.batch_size
            nsamples = min(local_batches, self.max_batches_online) * self.model.batch_size
            x_shape = [nsamples] + self.input_shape
            if self.model.tensor_format is PYDTNN_TENSOR_FORMAT.NHWC:
                x_shape = [x_shape[i] for i in (0, 2, 3, 1)]
            y_shape = [nsamples] + self.output_shape
            self._x[part] = np.zeros(x_shape, dtype=self.model.dtype, order="C")
            self._y[part] = np.zeros(y_shape, dtype=self.model.dtype, order="C")

    @abstractmethod
    def _init_actual_data(self):
        """Generates initial self._x[] and self._y[]. To be implemented in derived classes."""
        pass

    @staticmethod
    def _nchw2nhwc(x: Array) -> Array:
        return x.transpose(0, 2, 3, 1).copy()
    
    @staticmethod
    def _chw2hwc(x: Array) -> Array:
        return x.transpose(1, 2, 0).copy()
    
    @staticmethod
    def _hwc2chw(x: Array) -> Array:
        return x.transpose(2, 0, 1).copy()

    @staticmethod
    def _decode_class(y: Array, classes_list: np.ndarray) -> None:
        """Sets to 1 the corresponding entry in the 2D y array as indicated by the 1D array of classes"""
        y[np.arange(y.shape[0]), classes_list] = 1

    def _synthetic_data_generator(self, part: DatasetEnum):
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
        for p in (DatasetEnum.TRAIN, DatasetEnum.VAL, DatasetEnum.TEST):
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

    def _actual_data_generator(self, part: DatasetEnum) -> Generator[tuple[Array, Array]]:
        yield self._x[part], self._y[part]

    def _actual_batch_generator(self, part:DatasetEnum) -> Generator[tuple[Array, Array, int]]:
        # NOTE: global_batch_size should be MPI.reduce(x_local_batch.shape[0])
        # However to avoid communications per batch, we assume all process have our x_local_batch.shape[0]
        local_batch_size = self.model.batch_size
        global_batch_size = self.model.batch_size * self.model.nprocs
        generator = self._data_generator(part)
        nsamples = self._nsamples[part]
        for x_data, y_data in _BackgroundGenerator(generator):
            local_nsamples = x_data.shape[0]
            s = memoryview(np.arange(local_nsamples))
            if self.model.resize and not self.model.use_synthetic_data:
                x_data = self._do_resize(x_data)
            if part is DatasetEnum.TRAIN:
                np.random.shuffle(s)
                if not self.model.use_synthetic_data:
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
                global_batch_size = min(nsamples, global_batch_size)
                yield x_local_batch[:nsamples], y_local_batch[:nsamples], global_batch_size
                nsamples -= global_batch_size
            # Generate the last batch (with size < local_batch_size)
            last_batch_size = local_nsamples % local_batch_size
            if last_batch_size > 0:
                start = end
                end = start + last_batch_size  # = local_nsamples
                indices = s[start:end]
                x_local_batch = x_data[indices, ...]
                y_local_batch = y_data[indices, ...]
                global_batch_size = min(nsamples, global_batch_size)
                yield x_local_batch[:nsamples], y_local_batch[:nsamples], global_batch_size
                nsamples -= global_batch_size

    def _batch_generator(self, part:DatasetEnum) -> Generator[tuple[Array, Array, int]]:
        x_batch = np.zeros(shape=(0, *self.input_shape), dtype=self.model.dtype)
        y_batch = np.zeros(shape=(0, *self.output_shape), dtype=self.model.dtype)
        for x_batch, y_batch, batch_size in self._actual_batch_generator(part):
            yield x_batch, y_batch, batch_size
        # NOTE: The following infinite loop provides of empty batches 
        #        if there are asked more batches than actually are.
        while True:
            yield x_batch[:0], y_batch[:0], 0

    def _do_flip_images(self, data: Array) -> Array:
        match self.model.tensor_format:
            case PYDTNN_TENSOR_FORMAT.NCHW:
                n, c, h, w = data.shape
                width_dim = -1
            case PYDTNN_TENSOR_FORMAT.NHWC:
                n, h, w, c = data.shape
                width_dim = 2
            case _:
                raise NotImplementedError(f"\"Dataset _do_flip_image\" is not implemented for \"{self.model.tensor_format}\" format.")
            
        limit = min(n, int(n * self.model.flip_images_prob))
        s = np.arange(n)
        np.random.shuffle(s)
        s = s[:limit]
        data[s, ...] = np.flip(data[s, ...], axis=width_dim)
        return data

    def _do_crop_images(self, data: Array) -> Array:
        match self.model.tensor_format:
            case PYDTNN_TENSOR_FORMAT.NCHW:
                n, c, h, w = data.shape
            case PYDTNN_TENSOR_FORMAT.NHWC:
                n, h, w, c = data.shape
            case _:
                raise NotImplementedError(f"\"Dataset _do_crop_images\" is not implemented for \"{self.model.tensor_format}\" format.")
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
            match self.model.tensor_format:
                case PYDTNN_TENSOR_FORMAT.NCHW:
                    data[ri, :, :t[i], :ll[i]] = 0.0
                    data[ri, :, b:, r:] = 0.0
                case PYDTNN_TENSOR_FORMAT.NHWC:
                    data[ri, :t[i], :ll[i], :] = 0.0
                    data[ri, b:, r:, :] = 0.0
                case _:
                    raise NotImplementedError(f"\"Dataset _do_crop_images\" is not implemented for \"{self.model.tensor_format}\" format.")
            data[ri, ...] = np.roll(data[ri, ...], np.random.randint(-t[i], (h - b)), axis=1)
            data[ri, ...] = np.roll(data[ri, ...], np.random.randint(-ll[i], (w - r)), axis=2)
        return data
    
    def _do_resize(self, data: Array) -> Array:
        n = data.shape[0]
        new_data = np.empty(shape = (n, *self.resize_shape), dtype=self.model.dtype, order="C")
        for i in range(n):
            image = Image.fromarray(data[i], mode="RGB")
            # NOTE: resize: The requested size in pixels, as a tuple or array: (width, height), but we work with NC*HW* or N*HW*C ==> self.model.resize_dimension[::-1]
            resized_data = np.asarray(image.resize(self.model.resize_dimension[::-1]), dtype=self.model.dtype, order="C")
            if self.model.tensor_format is PYDTNN_TENSOR_FORMAT.NCHW:
                resized_data = self._hwc2chw(resized_data)
            # else: Do nothing, the resized_data.shape is correct.
            new_data[i] = resized_data
        return new_data
    # ---

    def _do_data_augmentation(self, x_data: Array) -> Array:
        # Preserve the original version when producing new data
        x_data = x_data.copy()
        if self.model.flip_images:
            x_data = self._do_flip_images(x_data)
        if self.model.crop_images:
            x_data = self._do_crop_images(x_data)
        return x_data
