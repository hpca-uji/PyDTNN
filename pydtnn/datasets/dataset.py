from pathlib import Path
import warnings
import itertools
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image
import rapidgzip

from pydtnn.utils.tensor import TensorFormat
from pydtnn.utils import BackgroundGenerator, string_substitute, random
from typing import TYPE_CHECKING, Generator, IO
if TYPE_CHECKING:
    from pydtnn.model import Model
from enum import IntEnum
from pydtnn.utils.types import Array, ArrayShape


class Dataset[T: Array](ABC):
    """
    NOTE
    - input_shape is expected to be in NCHW format
    - data_generator() is expected to be in model.dtype, normalized to [0, 1]
    - data_generator(x) is expected to be in model.tensor_format format
    - data_generator(y) is expected to be in NC format
    """

    class Part(IntEnum):
        TRAIN = 0
        VAL = 1
        TEST = 2

    def __init__(self, model: "Model", train_nsamples: int, test_nsamples: int, input_shape: ArrayShape,
                 output_shape: ArrayShape, force_test_as_validation=False, debug=False):

        if len(input_shape) != 3:
            warnings.warn(f"Input shape does not have 3 dimensions ({input_shape}), it may cause issues!", RuntimeWarning)
        # if len(input_shape) == 3 and not (input_shape[0] < input_shape[2]):
        elif not (input_shape[0] < input_shape[2]):
            warnings.warn(f"Dataset input_shape {input_shape} may not be in NCHW format, regardless of model format!", RuntimeWarning)

        if len(output_shape) != 1:
            warnings.warn(f"Output shape should have 1 dimension, but it has {len(output_shape)} (Output shape: {output_shape}). This may cause issues!", RuntimeWarning)

        self.model: Model = model
        self.debug: bool = debug
        self.test_as_validation: bool = self.model.test_as_validation or force_test_as_validation
        self._nsamples: list[int] = [train_nsamples, 0, test_nsamples]

        # Compute self._nsamples[DatasetEnum.VAL]
        if self.test_as_validation:
            self._nsamples[Dataset.Part.VAL] = self._nsamples[Dataset.Part.TEST]
        else:
            self._nsamples[Dataset.Part.VAL] = min(self._nsamples[Dataset.Part.TRAIN] - self.model.nprocs,
                                                   max(self.model.nprocs,
                                                       int(self._nsamples[Dataset.Part.TRAIN] * self.model.validation_split)))
            self._nsamples[Dataset.Part.TRAIN] -= self._nsamples[Dataset.Part.VAL]

        # self.real_input_shape = tuple(input_shape)
        self.input_shape = tuple(input_shape)
        self.output_shape = tuple(output_shape)

        if self.model.transform_crop:
            crop, size = self._calculate_crop(self.input_shape[1:])
            self.input_shape = (self.input_shape[0], *size)

        if self.model.transform_resize:
            self.input_shape = (self.input_shape[0], self.model.transform_resize_size, self.model.transform_resize_size)

        self._initial_nsamples = [self._nsamples[Dataset.Part.TRAIN], self._nsamples[Dataset.Part.VAL], self._nsamples[Dataset.Part.TEST]]
        # Offset (in number of samples) and number of samples for the current job for each dataset part
        self._local_offset = [0] * 3
        self._local_nsamples = [0] * 3
        self._local_remaining_nsamples = [-1] * 3  # -1 is used to mark each part as not initialized

        for part in Dataset.Part.TRAIN, Dataset.Part.VAL, Dataset.Part.TEST:
            (self._local_offset[part],
             self._local_nsamples[part],
             self._nsamples[part]
             ) = self._compute_local_workload(self._nsamples[part])

        # Declare _x and _y for train, val and test dataset parts
        self._x: list[np.ndarray]
        self._y: list[np.ndarray]

        if self.model.use_synthetic_data:
            self._data_generator = self._synthetic_data_generator
            self._init_synthetic_data()
        else:
            self._data_generator = self._actual_data_generator
            self._init_actual_data()

        self.x_empty_batch = np.zeros(shape=(0, *self.input_shape), dtype=self.model.dtype)
        self.y_empty_batch = np.zeros(shape=(0, *self.output_shape), dtype=self.model.dtype)

        if self.debug:
            self._print_report()

    def _gzip_open(self, filename: str) -> IO[bytes]:
        path = Path(filename)
        plain = path.with_suffix("")
        idx = path.with_suffix(f"{path.suffix}.idx")

        if plain.exists():
            return open(plain, mode="rb")
        try:
            f = rapidgzip.RapidgzipFile(path, parallelization=1)
            if idx.exists():
                f.import_index(str(idx))
            else:
                f.export_index(str(idx))
        except Exception:
            f.close()
            raise
        else:
            return f

    def export(self, split_weights: list[float] | None = None):
        """Export dataset (possibly split and rank specific)"""

        # Get split weights
        if split_weights is None:
            split_weights = list(map(float, self.model.dataset_export_split_weights.split(",")))

        # Data generators
        gen_train = self._data_generator(Dataset.Part.TRAIN)
        gen_val = self._data_generator(Dataset.Part.VAL)
        gen_test = self._data_generator(Dataset.Part.TEST)

        # Reconstruct validation split
        if self.test_as_validation:
            gen_test = itertools.chain(gen_test, gen_val)
        else:
            gen_train = itertools.chain(gen_train, gen_val)

        # T from generators
        x_train, y_train = map(np.concat, zip(*gen_train))
        x_test, y_test = map(np.concat, zip(*gen_test))

        # Ensure dataset is in NCHW
        match self.model.tensor_format:
            case TensorFormat.NCHW:
                pass
            case TensorFormat.NHWC:
                x_train = self._nhwc2nchw(x_train)
                x_test = self._nhwc2nchw(x_test)
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
            path = string_substitute(self.model.dataset_path, split=split)

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
        return self._nsamples[Dataset.Part.TRAIN]

    @property
    def val_nsamples(self):
        return self._nsamples[Dataset.Part.VAL]

    @property
    def test_nsamples(self):
        return self._nsamples[Dataset.Part.TEST]

    def get_train_val_generator(self) -> tuple[Generator[tuple[T, T, int]], Generator[tuple[T, T, int]]]:
        return (self._batch_generator(Dataset.Part.TRAIN),
                self._batch_generator(Dataset.Part.VAL))

    def get_test_generator(self) -> Generator[tuple[T, T, int]]:
        return self._batch_generator(Dataset.Part.TEST)

    def _print_report(self):
        if self.model.comm_rank == 0:
            print(f"Initial nsamples:"
                  f" train: {self._initial_nsamples[Dataset.Part.TRAIN]} "
                  f" val: {self._initial_nsamples[Dataset.Part.VAL]} "
                  f" test: {self._initial_nsamples[Dataset.Part.TEST]} "
                  )
        desc = ["train", "val", "test"]
        for part in (Dataset.Part.TRAIN, Dataset.Part.VAL, Dataset.Part.TEST):
            prefix = f"{self.model.rank}: " if part is Dataset.Part.TRAIN else "   "
            print(f"{prefix}"
                  f" {desc[part]} offset: {self._local_offset[part]}"
                  f" {desc[part]} local nsamples: {self._local_nsamples[part]}"
                  f" {desc[part]} nsamples: {self._nsamples[part]}"
                  )

    def _compute_local_workload(self, nsamples: int):
        """Computes the offset (in number of samples) and the number of samples for the current rank"""

        # Reduce nsamples according to steps per epoch
        global_batch_size = self.model.batch_size * self.model.nprocs
        batches_per_worker = nsamples / global_batch_size

        if self.model.dataset_percentage != 0:
            nsamples *= self.model.dataset_percentage

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

        return int(local_offset), int(local_nsamples), int(nsamples)

    def _init_synthetic_data(self):
        for part in Dataset.Part.TRAIN, Dataset.Part.VAL, Dataset.Part.TEST:
            local_batches = self._local_nsamples[part] // self.model.batch_size
            nsamples = local_batches * self.model.batch_size
            x_shape = (nsamples, *self.input_shape)
            if self.model.tensor_format is TensorFormat.NHWC:
                x_shape = tuple(x_shape[i] for i in (0, 2, 3, 1))
            y_shape = (nsamples, *self.output_shape)
            self._x[part] = np.zeros(x_shape, dtype=self.model.dtype, order="C")
            self._y[part] = np.zeros(y_shape, dtype=self.model.dtype, order="C")

    @abstractmethod
    def _init_actual_data(self):
        """Generates initial self._x[] and self._y[]. To be implemented in derived classes."""
        pass

    @staticmethod
    def _nchw2nhwc(x: T) -> T:
        return x.transpose(0, 2, 3, 1)

    @staticmethod
    def _nhwc2nchw(x: T) -> T:
        return x.transpose(0, 3, 1, 2)

    @staticmethod
    def _chw2hwc(x: T) -> T:
        return x.transpose(1, 2, 0)

    @staticmethod
    def _hwc2chw(x: T) -> T:
        return x.transpose(2, 0, 1)

    @staticmethod
    def _decode_class(y: T, classes_list: np.ndarray) -> None:
        """Sets to 1 the corresponding entry in the 2D y array as indicated by the 1D array of classes"""
        y[np.arange(y.shape[0]), classes_list] = 1

    def _synthetic_data_generator(self, part: Part):
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
        for p in (Dataset.Part.TRAIN, Dataset.Part.VAL, Dataset.Part.TEST):
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
    def _offset2files(filenames: list[str], images_per_file: int, local_offset: int, local_nsamples: int) -> list[tuple[str, int, int]]:
        i = local_offset // images_per_file
        offset_in_file = local_offset - i * images_per_file
        output = []
        while local_nsamples:
            nsamples = min(images_per_file - offset_in_file, local_nsamples)
            output.append((filenames[i], offset_in_file, nsamples))
            offset_in_file = 0
            local_nsamples -= nsamples
        return output

    def _actual_data_generator(self, part: Part) -> Generator[tuple[np.ndarray, np.ndarray]]:
        # NOTE: Yield is necessary so this function is a generator,
        #       however want to produce a empty iterable,
        #       so we return immediately without yielding
        return
        yield

    def _data_transform(self, part: Part, data: np.ndarray) -> np.ndarray:
        if self.model.use_synthetic_data:
            return data

        # NOTE: Don't modify data for the producer and ensure a mutable copy for transforms
        data = data.copy()

        if self.model.transform_crop:
            data = self._do_crop(data)

        if self.model.transform_resize:
            data = self._do_resize(data)

        if part is Dataset.Part.TRAIN:
            if self.model.augment_flip:
                data = self._do_flip_images(data)

            if self.model.augment_crop:
                data = self._do_crop_images(data)

            if self.model.augment_shuffle:
                random.shuffle(data)

        if self.model.normalize:
            data = self._do_normalize(data)

        return data

    def _actual_batch_generator(self, part: Part) -> Generator[tuple[np.ndarray, np.ndarray, int]]:
        # NOTE: global_batch_size should be MPI.reduce(x_local_batch.shape[0])
        # However to avoid communications per batch, we assume all process have our x_local_batch.shape[0]
        local_batch_size = self.model.batch_size
        global_batch_size = self.model.batch_size * self.model.nprocs

        def transform_generator():
            for x, y in self._data_generator(part):
                yield self._data_transform(part, x), y

        generator = transform_generator()
        nsamples = self._nsamples[part]

        batch_size = 0
        batch_online = []

        while nsamples > 0:
            for x_data, y_data in generator:
                batch_online.append((x_data, y_data))
                batch_size += x_data.shape[0]
                if batch_size >= local_batch_size:
                    break

            if batch_size <= 0:
                break

            x_data, y_data = zip(*batch_online)
            x_data = np.concatenate(x_data)
            y_data = np.concatenate(y_data)
            batch_online.clear()
            batch_size = 0

            x_data, x_extra = x_data[:local_batch_size], x_data[local_batch_size:]
            y_data, y_extra = y_data[:local_batch_size], y_data[local_batch_size:]
            if extra_size := x_extra.shape[0]:
                batch_online.append((x_extra, y_extra))
                batch_size += extra_size

            while x_data.shape[0] >= local_batch_size:
                x_batch, x_data = x_data[:local_batch_size], x_data[local_batch_size:]
                y_batch, y_data = y_data[:local_batch_size], y_data[local_batch_size:]

                global_batch_size = min(nsamples, global_batch_size)
                yield x_batch[:nsamples], y_batch[:nsamples], global_batch_size
                nsamples -= global_batch_size

    def _batch_generator(self, part: Part) -> Generator[tuple[np.ndarray, np.ndarray, int]]:
        yield from BackgroundGenerator(self._actual_batch_generator(part), max_prefetch=1)
        # NOTE: The following infinite loop provides of empty batches
        #        if there are asked more batches than actually are.
        while True:
            yield self.x_empty_batch, self.y_empty_batch, 0

    def _do_normalize(self, data: np.ndarray) -> np.ndarray:
        data += self.model.normalize_offset
        data *= self.model.normalize_scale
        return data

    def _do_flip_images(self, data: np.ndarray) -> np.ndarray:
        match self.model.tensor_format:
            case TensorFormat.NCHW:
                n, c, h, w = data.shape
                width_dim = -1
            case TensorFormat.NHWC:
                n, h, w, c = data.shape
                width_dim = 2
            case _:
                raise NotImplementedError(f"\"Dataset _do_flip_image\" is not implemented for \"{self.model.tensor_format}\" format.")

        limit = min(n, int(n * self.model.augment_flip_prob))
        s = np.arange(n)
        random.shuffle(s)
        s = s[:limit]
        data[s, ...] = np.flip(data[s, ...], axis=width_dim)
        return data

    def _do_crop_images(self, data: np.ndarray) -> np.ndarray:
        match self.model.tensor_format:
            case TensorFormat.NCHW:
                n, c, h, w = data.shape
            case TensorFormat.NHWC:
                n, h, w, c = data.shape
            case _:
                raise NotImplementedError(f"\"Dataset _do_crop_images\" is not implemented for \"{self.model.tensor_format}\" format.")
        crop_size = min(self.model.augment_crop_size, h, w)
        limit = min(n, int(n * self.model.augment_crop_prob))
        s = np.arange(n)
        random.shuffle(s)
        s = s[:limit]
        t = random.integers(0, h - crop_size, (limit,))
        ll = random.integers(0, w - crop_size, (limit,))
        for i, ri in enumerate(s):
            b, r = t[i] + crop_size, ll[i] + crop_size
            # batch[ri,...] = transform_resize(batch[ri,:,t[i]:b,l[i]:r], (ri.size,c,h,w))
            match self.model.tensor_format:
                case TensorFormat.NCHW:
                    data[ri, :, :t[i], :ll[i]] = 0.0
                    data[ri, :, b:, r:] = 0.0
                case TensorFormat.NHWC:
                    data[ri, :t[i], :ll[i], :] = 0.0
                    data[ri, b:, r:, :] = 0.0
                case _:
                    raise NotImplementedError(f"\"Dataset _do_crop_images\" is not implemented for \"{self.model.tensor_format}\" format.")
            data[ri, ...] = np.roll(data[ri, ...], random.integers(-t[i], (h - b)), axis=1)
            data[ri, ...] = np.roll(data[ri, ...], random.integers(-ll[i], (w - r)), axis=2)
        return data

    def _do_resize(self, data: np.ndarray) -> np.ndarray:
        match self.model.tensor_format:
            case TensorFormat.NHWC:
                data = self._nhwc2nchw(data)
            case TensorFormat.NCHW:
                pass
            case _:
                raise ValueError("Unsupported tensor format")

        size = (self.model.transform_resize_size, self.model.transform_resize_size)
        shape = (*data.shape[:2], *size)
        N, C, H, W = shape

        new_data = np.empty(shape=shape, dtype=self.model.dtype, order="C")

        for n in range(N):
            for c in range(C):
                channel = data[n, c]
                # NOTE: PIL mode F is WH in float32
                channel = channel.transpose().astype(np.float32)
                image = Image.fromarray(channel, mode="F")
                image = image.resize(size)
                channel = np.asarray(image, dtype=np.float32)
                channel = channel.transpose().astype(self.model.dtype)
                new_data[n, c] = channel

        match self.model.tensor_format:
            case TensorFormat.NHWC:
                new_data = self._nchw2nhwc(new_data)
            case TensorFormat.NCHW:
                pass
            case _:
                raise ValueError("Unsupported tensor format")

        return new_data
    # ---

    def _calculate_crop(self, size: tuple[int, int]) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
        width, height = size
        frame_fraction = (1 - self.model.transform_crop_perc) / 2
        x_offset, y_offset = round(width * frame_fraction), round(height * frame_fraction)
        crop = (x_offset, y_offset, width - x_offset, height - y_offset)
        size = (crop[2] - crop[0], crop[3] - crop[1])
        return (crop, size)

    def _do_crop(self, data: np.ndarray) -> np.ndarray:
        match self.model.tensor_format:
            case TensorFormat.NHWC:
                data = self._nhwc2nchw(data)
            case TensorFormat.NCHW:
                pass
            case _:
                raise ValueError("Unsupported tensor format")

        size = data.shape[2:4]
        crop, size = self._calculate_crop(size)
        shape = (*data.shape[:2], *size)
        N, C, H, W = shape

        new_data = np.empty(shape=shape, dtype=self.model.dtype, order="C")

        for n in range(N):
            for c in range(C):
                channel = data[n, c]
                # NOTE: PIL mode F is WH in float32
                channel = channel.transpose().astype(np.float32)
                image = Image.fromarray(channel, mode="F")
                image = image.crop(crop)
                channel = np.asarray(image, dtype=np.float32)
                channel = channel.transpose().astype(self.model.dtype)
                new_data[n, c] = channel

        match self.model.tensor_format:
            case TensorFormat.NHWC:
                new_data = self._nchw2nhwc(new_data)
            case TensorFormat.NCHW:
                pass
            case _:
                raise ValueError("Unsupported tensor format")

        return new_data
    # ---
    def _load_rgb_image(self, fp: IO[bytes] | str) -> np.ndarray:
        """Transform a file-like (RGB image) to array (ndarray CHW uint8)"""
        with Image.open(fp=fp) as image:
            image = image.convert("RGB")
            array = np.asarray(image)
            # NOTE: PIL mode RGB is WHC in unit8
            array = array.transpose(2, 1, 0)
        return array

    def _load_gray_image(self, fp: IO[bytes] | str) -> np.ndarray:
        """Transform a file-like (gray-scale image) to array (ndarray CHW uint8)"""
        with Image.open(fp=fp) as image:
            image = image.convert("L")
            array = np.asarray(image)
            # NOTE: PIL mode L is WH in unit8
            array = array.transpose(1, 0)
            array = array[None, ...]
        return array
    # ----
