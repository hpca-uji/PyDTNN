"""
Dataset module for PyDTNN.

Provides the base Dataset class and utility functions for managing,
transforming, and generating data batches for machine learning models.
"""

from __future__ import annotations

import functools
import itertools
import logging
import warnings
from pathlib import Path
from typing import IO, TYPE_CHECKING, Callable, Generator

import numpy as np
import rapidgzip
from PIL import Image

from pydtnn.datasets.abstract.base import Base
from pydtnn.utils import BackgroundGenerator
from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.tensor import ChannelFormat, SampleFormat, TensorFormat, format_transpose

__all__ = ("Init",)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pydtnn.model.utils import Utils as Model


type TransformFunc = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


class Init(Base):
    """
    Base class for handling datasets in PyDTNN.

    This class provides a framework for loading, transforming, and batching data
    for machine learning models. It supports various data augmentation techniques,
    normalization, and distributed data loading.

    NOTE
    - input_shape is expected to be in NCHW format
    - data_generator() is expected to be in model.dtype, normalized to [0, 1]
    - data_generator(x) is expected to be in model.tensor_format format
    - data_generator(y) is expected to be in NC format
    """

    def __init__(self, model: Model, train_nsamples: int = 0, test_nsamples: int = 0, input_shape: ArrayShape = (), output_shape: ArrayShape = (), force_test_as_validation=False, debug=False):
        """
        Initialize the dataset with model configuration and sample parameters.

        Args:
            model: The model instance, providing configuration like batch size,
                   dtype, tensor format, and data paths.
            train_nsamples: The total number of samples intended for training.
            test_nsamples: The total number of samples intended for testing.
            input_shape: The shape of a single input sample (excluding batch size),
                         expected in NCHW format (Channels, Height, Width).
            output_shape: The shape of a single output sample (excluding batch size).
            force_test_as_validation: If True, the test set will be used as the
                                      validation set.
            debug: If True, print detailed reports about dataset configuration
                   and workload distribution.
        """

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

        # Compute self._nsamples[DatasetEnum.VAL]
        if self.test_as_validation:
            self._nsamples[Base.Part.VAL] = self._nsamples[Base.Part.TEST]
        else:
            self._nsamples[Base.Part.VAL] = min(self._nsamples[Base.Part.TRAIN] - self.model.nprocs, max(self.model.nprocs, int(self._nsamples[Base.Part.TRAIN] * self.model.validation_split)))
            self._nsamples[Base.Part.TRAIN] -= self._nsamples[Base.Part.VAL]

        # self.real_input_shape = tuple(input_shape)
        self.input_shape: ArrayShape = tuple(input_shape)
        self.output_shape: ArrayShape = tuple(output_shape)

        self._initial_nsamples = [self._nsamples[Base.Part.TRAIN], self._nsamples[Base.Part.VAL], self._nsamples[Base.Part.TEST]]
        # Offset (in number of samples) and number of samples for the current job for each dataset part
        self._local_offset = [0] * len(Base.Part)
        self._local_nsamples = [0] * len(Base.Part)
        self._local_remaining_nsamples = [-1] * len(Base.Part)  # -1 is used to mark each part as not initialized

        for part in Base.Part.TRAIN, Base.Part.VAL, Base.Part.TEST:
            (self._local_offset[part], self._local_nsamples[part], self._nsamples[part]) = self._compute_local_workload(self._nsamples[part])

        self._data_generator = self._actual_data_generator

        if self.debug:
            self._print_report()

    def _gzip_open(self, filename: str) -> IO[bytes]:
        """
        Open a gZIP file.

        This method handles opening gZIP files, creating or loading their
        seek tables for efficient random access.

        Args:
            filename: The path to the gZIP file.

        Returns:
            A file-like object opened in binary read mode.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            Exception: If there's an error during rapidgzip file operations.
        """
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
        """
        Export dataset to a dictionary of numpy arrays.

        This method reconstructs the entire dataset (or specified partitions)
        into numpy arrays, which can then be saved or further processed.

        Returns:
            A dictionary containing the dataset splits ('x_train', 'y_train',
            'x_test', 'y_test') as numpy arrays.
        """

        # Data generators
        gen_train = BackgroundGenerator(self._actual_batch_generator(Base.Part.TRAIN), max_prefetch=1)
        gen_val = BackgroundGenerator(self._actual_batch_generator(Base.Part.VAL), max_prefetch=1)
        gen_test = BackgroundGenerator(self._actual_batch_generator(Base.Part.TEST), max_prefetch=1)
        num_train = self._local_nsamples[Base.Part.TRAIN]
        num_val = self._local_nsamples[Base.Part.VAL]
        num_test = self._local_nsamples[Base.Part.TEST]

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
            x_train[offset: offset + n] = self.model.decode_tensor(x_batch)
            y_train[offset: offset + n] = y_batch
            offset += n
        offset = 0
        for i, (x_batch, y_batch, _) in enumerate(gen_test):
            n = x_batch.shape[0]
            x_test[offset: offset + n] = self.model.decode_tensor(x_batch)
            y_test[offset: offset + n] = y_batch
            offset += n

        return {
            "name": self.name,  # type: ignore
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test,
        }

    def _export_split(self, data: dict[str, np.ndarray], split_weights: list[float] = [1.0]) -> Generator[dict[str, np.ndarray]]:
        """
        Generate export data splits based on weights.

        This method takes exported dataset data and splits it into multiple
        subsets according to the provided weights.

        Args:
            data: A dictionary containing the dataset splits ('x_train', 'y_train',
                  'x_test', 'y_test').
            split_weights: A list of weights defining how to split the data.
                           The sum of weights determines the total proportion.

        Yields:
            Dictionaries, each representing a split subset of the dataset.
        """

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
            yield {**data, "x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}

    def export_archive(self, path: Path | None = None, split_weights: list[float] | None = None):
        """
        Export dataset to an archive file.

        The dataset is exported to a compressed NumPy archive (.npz).
        Optionally, the dataset can be split into multiple archives based on weights.

        Args:
            path: The directory path where the archive(s) will be saved.
                  Defaults to `self.model.dataset_path`.
            split_weights: If provided, the dataset will be split into multiple
                           archives based on these weights.
        """
        data = self.export()
        path = path if path else Path(self.model.dataset_path)

        if split_weights:
            datas = self._export_split(data, split_weights)
            for split, data in enumerate(datas):
                np.savez_compressed(path / f"archive.{split}.npz", **data)  # type: ignore
        else:
            np.savez_compressed(path / "archive.npz", **data)  # type: ignore

    def get_train_val_generator(self) -> tuple[Generator[tuple[np.ndarray, np.ndarray, int]], Generator[tuple[np.ndarray, np.ndarray, int]]]:
        """
        Return generators for training and validation sets.

        These generators yield batches of data suitable for training and
        validation loops.

        Returns:
            A tuple containing two generators: (training_generator, validation_generator).
        """
        return (self._batch_generator(Base.Part.TRAIN), self._batch_generator(Base.Part.VAL))

    def get_test_generator(self) -> Generator[tuple[np.ndarray, np.ndarray, int]]:
        """
        Return generator for test set.

        This generator yields batches of data suitable for testing.

        Returns:
            A generator yielding test data batches.
        """
        return self._batch_generator(Base.Part.TEST)

    def _print_report(self):
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

    def _compute_local_workload(self, nsamples: int):
        """
        Computes the offset and number of samples for the current rank.

        This method distributes the total number of samples for a given partition
        among all available processes (ranks) based on the model's configuration
        (batch size, number of processes, steps per epoch, etc.).

        Args:
            nsamples: The total number of samples for the partition.

        Returns:
            A tuple containing:
            - local_offset: The starting index of samples for this rank.
            - local_nsamples: The number of samples assigned to this rank.
            - nsamples: The effective total number of samples after adjustments
                        (e.g., for steps_per_epoch).
        """

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

    def _model_init(self):
        """Generates initial self._x[] and self._y[]. To be implemented in derived classes."""
        self.x_empty_batch = np.zeros(shape=self.model.encode_shape((0, *self.input_shape)), dtype=self.model.dtype)
        self.y_empty_batch = np.zeros(shape=(0, *self.output_shape), dtype=self.model.dtype)

        # Declare _x and _y for train, val and test dataset parts
        self._x = [self.x_empty_batch] * len(Base.Part)
        self._y = [self.y_empty_batch] * len(Base.Part)

    @staticmethod
    def _nchw2nhwc(x: np.ndarray) -> np.ndarray:
        """Convert NCHW tensor to NHWC."""
        return format_transpose(x, TensorFormat.NCHW, TensorFormat.NHWC)

    @staticmethod
    def _nhwc2nchw(x: np.ndarray) -> np.ndarray:
        """Convert NHWC tensor to NCHW."""
        return format_transpose(x, TensorFormat.NHWC, TensorFormat.NCHW)

    @staticmethod
    def _chw2hwc(x: np.ndarray) -> np.ndarray:
        """Convert CHW sample to HWC."""
        return format_transpose(x, SampleFormat.CHW, SampleFormat.HWC)

    @staticmethod
    def _hwc2chw(x: np.ndarray) -> np.ndarray:
        """Convert HWC sample to CHW."""
        return format_transpose(x, SampleFormat.HWC, SampleFormat.CHW)

    @staticmethod
    def _decode_class(y: np.ndarray, classes_list: np.ndarray) -> None:
        """Sets to 1 the corresponding entry in the 2D y array as indicated by the 1D array of classes"""
        y[np.arange(y.shape[0]), classes_list] = 1

    @staticmethod
    def _offset2files(filenames: list[str], images_per_file: int, local_offset: int, local_nsamples: int) -> list[tuple[str, int, int]]:
        """
        Map local offset and sample count to specific files.

        Given a list of filenames, the number of samples per file, a local offset,
        and a local sample count, this method determines which files and which
        ranges within those files contain the required samples.

        Args:
            filenames: A list of filenames.
            images_per_file: The number of samples contained in each file.
            local_offset: The starting sample index for the current request.
            local_nsamples: The total number of samples to retrieve.

        Returns:
            A list of tuples, where each tuple contains:
            (filename, offset_in_file, num_samples_from_file).
        """
        i = local_offset // images_per_file
        offset_in_file = local_offset - i * images_per_file
        output = []
        while local_nsamples:
            nsamples = min(images_per_file - offset_in_file, local_nsamples)
            output.append((filenames[i], offset_in_file, nsamples))
            offset_in_file = 0
            local_nsamples -= nsamples
        return output

    def _actual_data_generator(self, part: Base.Part) -> Generator[tuple[np.ndarray, np.ndarray]]:
        """Yield raw data from the dataset partition."""
        yield self._x[part], self._y[part]

    @staticmethod
    def _x_transformer_adaptor(func: Callable[[np.ndarray], np.ndarray]) -> TransformFunc:
        """
        Adapt a single-input transformation function to the (x, y) signature.

        This utility function wraps a transformation that operates only on the
        input data (x) so that it can be used within the dataset's transformation
        pipeline, which expects functions that take (x, y) and return (x, y).

        Args:
            func: The transformation function that takes a single numpy array (x)
                  and returns a transformed numpy array.

        Returns:
            A new function that accepts (x, y) and applies `func` to `x`,
            returning the transformed `x` and the original `y`.
        """

        @functools.wraps(func)
        def wrapper(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return func(x), y

        return wrapper

    def _base_data_generator(self, part: Base.Part) -> Generator[tuple[np.ndarray, np.ndarray]]:
        """
        Yield transformed data from the dataset partition.

        This generator applies all registered transformations for a given dataset
        partition to the raw data before yielding it.

        Args:
            part: The dataset partition (TRAIN, VAL, or TEST) to generate data from.

        Yields:
            Tuples of transformed input (x) and output (y) data.
        """
        for x, y in self._data_generator(part):
            x, y = x.copy(), y.copy()
            yield x, y

    def _actual_batch_generator(self, part: Base.Part) -> Generator[tuple[np.ndarray, np.ndarray, int]]:
        """
        Generate batches of data for the specified partition.

        This method orchestrates the creation of batches from the transformed data.
        It handles accumulating samples into batches of the specified size and
        yields them along with the effective global batch size for that yield.

        Args:
            part: The dataset partition (TRAIN, VAL, or TEST) to generate batches from.

        Yields:
            Tuples of (x_batch, y_batch, effective_global_batch_size).
        """

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

        generator = self._base_data_generator(part)
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

            while (x_data.shape[0] > 0) and ((x_data.shape[0] >= local_batch_size) or (local_batch_size >= nsamples)):
                x_batch, x_data = x_data[:local_batch_size], x_data[local_batch_size:]
                y_batch, y_data = y_data[:local_batch_size], y_data[local_batch_size:]

                global_batch_size = min(nsamples, global_batch_size)
                yield x_batch[:nsamples], y_batch[:nsamples], global_batch_size
                nsamples -= global_batch_size

    def _batch_generator(self, part: Base.Part) -> Generator[tuple[np.ndarray, np.ndarray, int]]:
        """
        Yield batches with background prefetching.

        This method wraps the actual batch generator with a `BackgroundGenerator`
        to enable prefetching of batches, improving data loading performance.

        Args:
            part: The dataset partition (TRAIN, VAL, or TEST) to generate batches from.

        Yields:
            Tuples of (x_batch, y_batch, effective_global_batch_size), prefetched.
        """
        yield from BackgroundGenerator(self._actual_batch_generator(part), max_prefetch=1)

        # NOTE: The following infinite loop provides of empty batches
        #       if there are asked more batches than actually are.
        x_empty_batch = np.zeros(shape=self.model.encode_shape((0, *self.input_shape)), dtype=self.model.dtype)
        y_empty_batch = np.zeros(shape=(0, *self.output_shape), dtype=self.model.dtype)
        while True:
            yield x_empty_batch, y_empty_batch, 0

    def _load_rgb_image(self, fp: IO[bytes] | str) -> np.ndarray:
        """
        Transform a file-like object (RGB image) to a numpy array.

        Opens an image file, converts it to RGB format, and returns it as a
        numpy array with shape (C, H, W) and dtype uint8.

        Args:
            fp: A file-like object or a string path to the image file.

        Returns:
            A numpy array representing the RGB image in CHW format.
        """
        with Image.open(fp=fp) as image:
            image = image.convert("RGB")
            array = np.asarray(image, order="C")
            # NOTE: PIL mode RGB is WHC in unit8
            array = format_transpose(array, SampleFormat.WHC, SampleFormat.CHW)
        return array

    def _load_gray_image(self, fp: IO[bytes] | str) -> np.ndarray:
        """
        Transform a file-like object (gray-scale image) to a numpy array.

        Opens an image file, converts it to grayscale ('L' mode), and returns
        it as a numpy array with shape (1, H, W) and dtype uint8.

        Args:
            fp: A file-like object or a string path to the image file.

        Returns:
            A numpy array representing the grayscale image in CHW format.
        """
        with Image.open(fp=fp) as image:
            image = image.convert("L")
            array = np.asarray(image, order="C")
            # NOTE: PIL mode L is WH in unit8
            array = format_transpose(array, ChannelFormat.WH, ChannelFormat.HW)
            array = array[None, ...]
        return array
