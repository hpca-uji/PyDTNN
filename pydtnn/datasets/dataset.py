from __future__ import annotations

import functools
import itertools
import logging
import warnings
from enum import IntEnum
from pathlib import Path
from typing import IO, TYPE_CHECKING, Callable, Generator

import numpy as np
import rapidgzip
from PIL import Image

from pydtnn.utils import BackgroundGenerator, find_component, random
from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.tensor import (ChannelFormat, SampleFormat, TensorFormat,
                                 format_transpose)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pydtnn.model.utils import Utils as Model


type TransformFunc = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


class Dataset:
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

    def __init__(self, model: Model, train_nsamples: int = 0, test_nsamples: int = 0, input_shape: ArrayShape = (),
                 output_shape: ArrayShape = (), force_test_as_validation=False, debug=False):

        if train_nsamples <= 0:
            raise ValueError("Dataset has no training samples!")
        elif test_nsamples <= 0:
            raise ValueError("Dataset has no test samples!")
        elif len(input_shape) <= 0:
            raise ValueError("Dataset has no input shape!")
        elif len(output_shape) <= 0:
            raise ValueError("Dataset has no output shape!")

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

        self._transformations = dict[Dataset.Part, list[TransformFunc]]()
        transformations_training = list[TransformFunc]()
        transformations_always = list[TransformFunc]()

        # Compute self._nsamples[DatasetEnum.VAL]
        if self.test_as_validation:
            self._nsamples[Dataset.Part.VAL] = self._nsamples[Dataset.Part.TEST]
        else:
            self._nsamples[Dataset.Part.VAL] = min(self._nsamples[Dataset.Part.TRAIN] - self.model.nprocs,
                                                   max(self.model.nprocs,
                                                       int(self._nsamples[Dataset.Part.TRAIN] * self.model.validation_split)))
            self._nsamples[Dataset.Part.TRAIN] -= self._nsamples[Dataset.Part.VAL]

        # self.real_input_shape = tuple(input_shape)
        self.input_shape: ArrayShape = tuple(input_shape)
        self.output_shape: ArrayShape = tuple(output_shape)

        if self.model.transform_crop:
            crop, size = self._calculate_crop(self.input_shape[1:])  # type: ignore (The cropped input shape will be a tuple[int, int])
            self.input_shape = (self.input_shape[0], *size)
            transformations_training.append(self._x_transformer_adaptor(self._do_transform_crop))
            transformations_always.append(self._x_transformer_adaptor(self._do_transform_crop))

        if self.model.transform_resize:
            self.input_shape = (self.input_shape[0], self.model.transform_resize_size, self.model.transform_resize_size)
            transformations_training.append(self._x_transformer_adaptor(self._do_transform_resize))
            transformations_always.append(self._x_transformer_adaptor(self._do_transform_resize))

        if self.model.augment_flip > 0:
            transformations_training.append(self._x_transformer_adaptor(self._do_flip_images))

        if self.model.augment_crop > 0:
            transformations_training.append(self._x_transformer_adaptor(self._do_augment_crop))

        if self.model.augment_shuffle:
            transformations_training.append(self._do_augment_shuffle)

        if self.model.normalize:
            transformations_training.append(self._x_transformer_adaptor(self._do_normalize))
            transformations_always.append(self._x_transformer_adaptor(self._do_normalize))

        self._transformations[Dataset.Part.TRAIN] = transformations_training
        self._transformations[Dataset.Part.TEST] = transformations_always
        self._transformations[Dataset.Part.VAL] = transformations_always

        self._initial_nsamples = [self._nsamples[Dataset.Part.TRAIN], self._nsamples[Dataset.Part.VAL], self._nsamples[Dataset.Part.TEST]]
        # Offset (in number of samples) and number of samples for the current job for each dataset part
        self._local_offset = [0] * len(Dataset.Part)
        self._local_nsamples = [0] * len(Dataset.Part)
        self._local_remaining_nsamples = [-1] * len(Dataset.Part)  # -1 is used to mark each part as not initialized

        for part in Dataset.Part.TRAIN, Dataset.Part.VAL, Dataset.Part.TEST:
            (self._local_offset[part],
             self._local_nsamples[part],
             self._nsamples[part]
             ) = self._compute_local_workload(self._nsamples[part])

        self.x_empty_batch = np.zeros(shape=self.model.encode_shape((0, *self.input_shape)), dtype=self.model.dtype)
        self.y_empty_batch = np.zeros(shape=(0, *self.output_shape), dtype=self.model.dtype)

        # Declare _x and _y for train, val and test dataset parts
        self._x = [self.x_empty_batch] * len(Dataset.Part)
        self._y = [self.y_empty_batch] * len(Dataset.Part)

        self._data_generator = self._actual_data_generator
        self._init_actual_data()

        if self.debug:
            self._print_report()

    @property
    def name(self) -> str:
        return type(self).__name__

    def _gzip_open(self, filename: str) -> IO[bytes]:
        """Open a gZIP file (creating or loading seek table)"""
        path = Path(filename)
        plain = path.with_suffix("")
        idx = path.with_suffix(f"{path.suffix}.idx")
        f = None

        if plain.exists():
            return open(plain, mode="rb")
        elif not path.exists():
            raise FileNotFoundError(path)
        try:
            f = rapidgzip.RapidgzipFile(path, parallelization=1)
            if idx.exists():
                f.import_index(str(idx))
            else:
                f.export_index(str(idx))
        except Exception:
            if f:
                f.close()
            raise
        else:
            return f

    def export(self) -> dict[str, np.ndarray]:
        """Export dataset"""

        # Data generators
        gen_train = BackgroundGenerator(self._actual_batch_generator(Dataset.Part.TRAIN), max_prefetch=1)
        gen_val = BackgroundGenerator(self._actual_batch_generator(Dataset.Part.VAL), max_prefetch=1)
        gen_test = BackgroundGenerator(self._actual_batch_generator(Dataset.Part.TEST), max_prefetch=1)
        num_train = self._local_nsamples[Dataset.Part.TRAIN]
        num_val = self._local_nsamples[Dataset.Part.VAL]
        num_test = self._local_nsamples[Dataset.Part.TEST]

        # Reconstruct validation split
        if not self.test_as_validation:
            gen_train = itertools.chain(gen_train, gen_val)
            num_train += num_val

        # Allocate data
        x_train = np.zeros((num_train, *self.input_shape), dtype=np.float64)
        y_train = np.zeros((num_train, *self.output_shape), dtype=np.float64)
        x_test = np.zeros((num_test, *self.input_shape), dtype=np.float64)
        y_test = np.zeros((num_test, *self.output_shape), dtype=np.float64)

        # Populate data
        offset = 0
        for i, (x_batch, y_batch, _) in enumerate(gen_train):
            n = x_batch.shape[0]
            x_train[offset:offset + n] = self.model.decode_tensor(x_batch)
            y_train[offset:offset + n] = y_batch
            offset += n
        offset = 0
        for i, (x_batch, y_batch, _) in enumerate(gen_test):
            n = x_batch.shape[0]
            x_test[offset:offset + n] = self.model.decode_tensor(x_batch)
            y_test[offset:offset + n] = y_batch
            offset += n

        return {
            "name": self.name,  # type: ignore
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test
        }

    def _export_split(self, data: dict[str, np.ndarray], split_weights: list[float] = [1]) -> Generator[dict[str, np.ndarray]]:
        """Generate export data splits"""

        # Get data
        x_train = data["x_train"]
        y_train = data["y_train"]
        x_test = data["x_test"]
        y_test = data["y_test"]

        # Calculate percentage splits
        total = sum(split_weights)
        split_percentage = [weight / total for weight in itertools.accumulate(split_weights)]

        # Split arrays
        np_splits = np.array(split_percentage[:-1])
        x_train = np.split(x_train, (len(x_train) * np_splits).astype(int))
        y_train = np.split(y_train, (len(y_train) * np_splits).astype(int))
        x_test = np.split(x_test, (len(x_test) * np_splits).astype(int))
        y_test = np.split(y_test, (len(y_test) * np_splits).astype(int))

        # Yield splits
        for x_train, y_train, x_test, y_test in zip(x_train, y_train, x_test, y_test):
            yield {
                **data,
                "x_train": x_train,
                "y_train": y_train,
                "x_test": x_test,
                "y_test": y_test
            }

    def export_archive(self, path: Path | None = None, split_weights: list[float] | None = None):
        """Export dataset to an archive"""
        data = self.export()
        path = path if path else Path(self.model.dataset_path)

        if split_weights:
            datas = self._export_split(data, split_weights)
            for split, data in enumerate(datas):
                np.savez_compressed(path / f"archive.{split}.npz", **data)  # type: ignore
        else:
            np.savez_compressed(path / "archive.npz", **data)  # type: ignore

    @property
    def train_nsamples(self):
        return self._nsamples[Dataset.Part.TRAIN]

    @train_nsamples.setter
    def train_nsamples(self, value):
        self._nsamples[Dataset.Part.TRAIN] = value

    @property
    def val_nsamples(self):
        return self._nsamples[Dataset.Part.VAL]

    @val_nsamples.setter
    def val_nsamples(self, value):
        self._nsamples[Dataset.Part.VAL] = value

    @property
    def test_nsamples(self):
        return self._nsamples[Dataset.Part.TEST]

    @test_nsamples.setter
    def test_nsamples(self, value):
        self._nsamples[Dataset.Part.TEST] = value

    def get_train_val_generator(self) -> tuple[Generator[tuple[np.ndarray, np.ndarray, int]], Generator[tuple[np.ndarray, np.ndarray, int]]]:
        return (self._batch_generator(Dataset.Part.TRAIN),
                self._batch_generator(Dataset.Part.VAL))

    def get_test_generator(self) -> Generator[tuple[np.ndarray, np.ndarray, int]]:
        return self._batch_generator(Dataset.Part.TEST)

    def _print_report(self):
        report = list[str]()
        if self.model.comm_rank == 0:
            report.append(f"Initial nsamples:")
            report.append(f" train: {self._initial_nsamples[Dataset.Part.TRAIN]} ")
            report.append(f" val: {self._initial_nsamples[Dataset.Part.VAL]} ")
            report.append(f" test: {self._initial_nsamples[Dataset.Part.TEST]} ")

        desc = ["train", "val", "test"]
        for part in (Dataset.Part.TRAIN, Dataset.Part.VAL, Dataset.Part.TEST):
            prefix = f"{self.model.rank}: " if part is Dataset.Part.TRAIN else "   "
            report.append(f"{prefix}")
            report.append(f" {desc[part]} offset: {self._local_offset[part]}")
            report.append(f" {desc[part]} local nsamples: {self._local_nsamples[part]}")
            report.append(f" {desc[part]} nsamples: {self._nsamples[part]}")

        logger.info('\n'.join(report))

    def _compute_local_workload(self, nsamples: int):
        """Computes the offset (in number of samples) and the number of samples for the current rank"""

        # Reduce nsamples according to steps per epoch
        global_batch_size = self.model.batch_size * self.model.nprocs
        batches_per_worker = nsamples / global_batch_size

        if self.model.dataset_percentage != 0:
            nsamples = nsamples * self.model.dataset_percentage  # type: ignore (It's expected to receive a int as parameter and it's fine like this)

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

    def _init_actual_data(self):
        """Generates initial self._x[] and self._y[]. To be implemented in derived classes."""
        pass

    @staticmethod
    def _nchw2nhwc(x: np.ndarray) -> np.ndarray:
        return format_transpose(x, TensorFormat.NCHW, TensorFormat.NHWC)

    @staticmethod
    def _nhwc2nchw(x: np.ndarray) -> np.ndarray:
        return format_transpose(x, TensorFormat.NHWC, TensorFormat.NCHW)

    @staticmethod
    def _chw2hwc(x: np.ndarray) -> np.ndarray:
        return format_transpose(x, SampleFormat.CHW, SampleFormat.HWC)

    @staticmethod
    def _hwc2chw(x: np.ndarray) -> np.ndarray:
        return format_transpose(x, SampleFormat.HWC, SampleFormat.CHW)

    @staticmethod
    def _decode_class(y: np.ndarray, classes_list: np.ndarray) -> None:
        """Sets to 1 the corresponding entry in the 2D y array as indicated by the 1D array of classes"""
        y[np.arange(y.shape[0]), classes_list] = 1

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
        yield self._x[part], self._y[part]

    @staticmethod
    def _x_transformer_adaptor(func: Callable[[np.ndarray], np.ndarray]) -> TransformFunc:
        @functools.wraps(func)
        def wrapper(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return func(x), y
        return wrapper

    def _transform_data_generator(self, part: Part) -> Generator[tuple[np.ndarray, np.ndarray]]:
        for x, y in self._data_generator(part):
            x, y = x.copy(), y.copy()
            for transformation in self._transformations[part]:
                x, y = transformation(x, y)
            yield x, y

    def _actual_batch_generator(self, part: Part) -> Generator[tuple[np.ndarray, np.ndarray, int]]:
        # NOTE: global_batch_size should be MPI.reduce(x_local_batch.shape[0])
        # However to avoid communications per batch, we assume all process have our x_local_batch.shape[0]

        # NOTE:
        # batch_size: en memoria
        # local_batch_size: a usar en esta iteración.

        # Casos:
        # -> El generador ha devuelto más datos que los que se necesita (es decir, que batch_size >= local_batch_size)
        #   ==> Se tienen que hacer un corte y guardarnos el restante para la sigueinte
        # -> El generador ha devuelto menos datos que los que se necesita (es decir, que batch_size < local_batch_size)
        #    * Queda dataset:
        #       ==> Guardarnos los datos y usarlos en la siguiente iteración.
        #    * No queda dataset:
        #       ==> Devolver lo que tengamos

        local_batch_size = self.model.batch_size
        global_batch_size = self.model.batch_size * self.model.nprocs

        generator = self._transform_data_generator(part)
        nsamples = self._nsamples[part]

        batch_size = 0
        batch_online = []

        while nsamples > 0:
            for x_data, y_data in generator:
                batch_online.append((x_data, y_data))
                batch_size += x_data.shape[0]
                if batch_size >= local_batch_size:
                    break
                    # Quedan más datos, pero tenemos suficientes ==> continuamos fuera del for.
                # else: Quedan datos, pero aún no hemos llenado el batch.
            # else (del for): # No queda más dataset

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

            # while (tenemos datos) and ((tenemos un batch completo) or (es el ultimo batch del dataset)):
            while (x_data.shape[0] > 0) and ((x_data.shape[0] >= local_batch_size) or (local_batch_size >= nsamples)):
                x_batch, x_data = x_data[:local_batch_size], x_data[local_batch_size:]
                y_batch, y_data = y_data[:local_batch_size], y_data[local_batch_size:]

                global_batch_size = min(nsamples, global_batch_size)
                yield x_batch[:nsamples], y_batch[:nsamples], global_batch_size
                nsamples -= global_batch_size

    def _batch_generator(self, part: Part) -> Generator[tuple[np.ndarray, np.ndarray, int]]:
        yield from BackgroundGenerator(self._actual_batch_generator(part), max_prefetch=1)
        # NOTE: The following infinite loop provides of empty batches
        #       if there are asked more batches than actually are.
        while True:
            yield self.x_empty_batch, self.y_empty_batch, 0

    def _do_normalize(self, data: np.ndarray) -> np.ndarray:
        np.add(data, self.model.normalize_offset, out=data)
        np.multiply(data, self.model.normalize_scale, out=data)
        return data

    def _do_flip_images(self, data: np.ndarray) -> np.ndarray:
        n = data.shape[0]
        match self.model.tensor_format:
            case TensorFormat.NCHW:
                width_dim = -1
            case TensorFormat.NHWC:
                width_dim = 2
            case _:
                raise NotImplementedError(f"Dataset _do_flip_image is not implemented for {self.model.tensor_format} format.")

        limit = min(n, int(n * self.model.augment_flip))
        s = np.arange(n)
        random.shuffle(s)
        s = s[:limit]
        data[s, ...] = np.flip(data[s, ...], axis=width_dim)
        return data

    def _do_augment_shuffle(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx = np.arange(x.shape[0])
        random.shuffle(idx)
        x[:] = x[idx]
        y[:] = y[idx]
        return x, y

    def _do_augment_crop(self, data: np.ndarray) -> np.ndarray:
        n, c, h, w = self.model.decode_shape(data.shape)
        crop_size = min(self.model.augment_crop_size, h, w)
        limit = min(n, int(n * self.model.augment_crop))
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
                    raise NotImplementedError(f"Dataset _do_crop_images is not implemented for {self.model.tensor_format} format.")
            data[ri, ...] = np.roll(data[ri, ...], random.integers(-t[i], (h - b)), axis=1)
            data[ri, ...] = np.roll(data[ri, ...], random.integers(-ll[i], (w - r)), axis=2)
        return data

    def _do_transform_resize(self, data: np.ndarray) -> np.ndarray:
        data = self.model.decode_tensor(data)

        size = (self.model.transform_resize_size, self.model.transform_resize_size)
        shape = (*data.shape[:2], *size)
        N, C, H, W = shape

        new_data = np.empty(shape=shape, dtype=self.model.dtype)

        for n in range(N):
            for c in range(C):
                channel: np.ndarray = data[n, c]
                # NOTE: PIL mode F is WH in float32
                channel = channel.transpose().astype(np.float32)  # type: ignore (it's possible to use copy=None)
                image = Image.fromarray(channel, mode="F")
                image = image.resize(size)
                channel = np.asarray(image, dtype=np.float32, order="C")
                channel = channel.transpose().astype(self.model.dtype)  # type: ignore (it's possible to use copy=None)
                new_data[n, c] = channel

        new_data = self.model.encode_tensor(new_data)

        return new_data

    def _calculate_crop(self, size: tuple[int, int]) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
        width, height = size
        frame_fraction = (1 - self.model.transform_crop_perc) / 2
        x_offset, y_offset = round(width * frame_fraction), round(height * frame_fraction)
        crop = (x_offset, y_offset, width - x_offset, height - y_offset)
        size = (crop[2] - crop[0], crop[3] - crop[1])
        return (crop, size)

    def _do_transform_crop(self, data: np.ndarray) -> np.ndarray:
        data = self.model.decode_tensor(data)

        size = data.shape[2:4]
        crop, size = self._calculate_crop(size)
        shape = (*data.shape[:2], *size)
        N, C, H, W = shape

        new_data = np.empty(shape=shape, dtype=self.model.dtype)

        for n in range(N):
            for c in range(C):
                channel: np.ndarray = data[n, c]
                # NOTE: PIL mode F is WH in float32
                channel = channel.transpose().astype(np.float32)  # type: ignore (it's possible to use copy=None)
                image = Image.fromarray(channel, mode="F")
                image = image.crop(crop)
                channel = np.asarray(image, dtype=np.float32, order="C")
                channel = channel.transpose().astype(self.model.dtype)  # type: ignore (it's possible to use copy=None)
                new_data[n, c] = channel

        new_data = self.model.encode_tensor(new_data)

        return new_data

    def _load_rgb_image(self, fp: IO[bytes] | str) -> np.ndarray:
        """Transform a file-like (RGB image) to array (ndarray CHW uint8)"""
        with Image.open(fp=fp) as image:
            image = image.convert("RGB")
            array = np.asarray(image, order="C")
            # NOTE: PIL mode RGB is WHC in unit8
            array = format_transpose(array, SampleFormat.WHC, SampleFormat.CHW)
        return array

    def _load_gray_image(self, fp: IO[bytes] | str) -> np.ndarray:
        """Transform a file-like (gray-scale image) to array (ndarray CHW uint8)"""
        with Image.open(fp=fp) as image:
            image = image.convert("L")
            array = np.asarray(image, order="C")
            # NOTE: PIL mode L is WH in unit8
            array = format_transpose(array, ChannelFormat.WH, ChannelFormat.HW)
            array = array[None, ...]
        return array


def select(name: str) -> type[Dataset]:
    assert __package__, "Package not found!"
    return find_component(__package__, name)
