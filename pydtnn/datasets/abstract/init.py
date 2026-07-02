"""
Dataset module for PyDTNN.

Provides the base Dataset class and utility functions for managing,
transforming, and generating data batches for machine learning models.
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Callable, Generator

import numpy as np

from pydtnn.datasets.abstract.base import Base
from pydtnn.datasets.abstract.utils import Utils
from pydtnn.utils import BackgroundGenerator
from pydtnn.utils.constants import ArrayShape

__all__ = ("Init",)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pydtnn.model.utils import Utils as Model


type TransformFunc = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


class Init(Utils):
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

    def __init__(
        self,
        model: Model,
        train_nsamples: int = 0,
        test_nsamples: int = 0,
        input_shape: ArrayShape = (),
        output_shape: ArrayShape = (),
        force_test_as_validation: bool = False,
        debug: bool = False,
    ) -> None:
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
        super().__init__()

        if train_nsamples <= 0:
            raise ValueError("Dataset has no training samples!")
        elif test_nsamples <= 0:
            raise ValueError("Dataset has no test samples!")
        elif len(input_shape) <= 0:
            raise ValueError("Dataset has no input shape!")
        elif len(output_shape) <= 0:
            raise ValueError("Dataset has no output shape!")

        if len(input_shape) != 3:
            logger.warning(
                f"Input shape does not have 3 dimensions ({input_shape}), it may cause issues!"
            )
        # if len(input_shape) == 3 and not (input_shape[0] < input_shape[2]):
        elif not (input_shape[0] < input_shape[2]):
            logger.warning(
                f"Dataset input_shape {input_shape} may not be in NCHW format, regardless of model"
                " format!"
            )

        if len(output_shape) != 1:
            logger.warning(
                f"Output shape should have 1 dimension, but it has {
                    len(output_shape)
                } (Output shape: {output_shape}). This may cause issues!"
            )

        self.model: Model = model
        self.debug: bool = debug
        self.test_as_validation: bool = self.model.test_as_validation or force_test_as_validation
        self._nsamples: list[int] = [train_nsamples, 0, test_nsamples]

        # Compute self._nsamples[DatasetEnum.VAL]
        if self.test_as_validation:
            self._nsamples[Base.Part.VAL] = self._nsamples[Base.Part.TEST]
        else:
            self._nsamples[Base.Part.VAL] = min(
                self._nsamples[Base.Part.TRAIN] - self.model.nprocs,
                max(
                    self.model.nprocs,
                    int(self._nsamples[Base.Part.TRAIN] * self.model.validation_split),
                ),
            )
            self._nsamples[Base.Part.TRAIN] -= self._nsamples[Base.Part.VAL]

        # self.real_input_shape = tuple(input_shape)
        self.input_shape: ArrayShape = tuple(input_shape)
        self.output_shape: ArrayShape = tuple(output_shape)

        self._initial_nsamples = [
            self._nsamples[Base.Part.TRAIN],
            self._nsamples[Base.Part.VAL],
            self._nsamples[Base.Part.TEST],
        ]
        # Offset (in number of samples) and number of samples for the current job
        # for each dataset part
        self._local_offset = [0] * len(Base.Part)
        self._local_nsamples = [0] * len(Base.Part)
        self._local_remaining_nsamples = [-1] * len(
            Base.Part
        )  # -1 is used to mark each part as not initialized

        for part in Base.Part.TRAIN, Base.Part.VAL, Base.Part.TEST:
            self._local_offset[part], self._local_nsamples[part], self._nsamples[part] = (
                self._compute_local_workload(self._nsamples[part])
            )

        if not self.test_as_validation:
            self._local_offset[Base.Part.VAL] += self._initial_nsamples[Base.Part.TRAIN]
        # self._local_offset[Base.Part.TEST] += self._initial_nsamples[Base.Part.TRAIN]

    def get_train_val_generator(
        self,
    ) -> tuple[
        Generator[tuple[np.ndarray, np.ndarray, int]], Generator[tuple[np.ndarray, np.ndarray, int]]
    ]:
        """
        Return generators for training and validation sets.

        These generators yield batches of data suitable for training and
        validation loops.

        Returns:
            A tuple containing two generators: (training_generator, validation_generator).
        """
        return (
            self._get_batch_generator(Base.Part.TRAIN),
            self._get_batch_generator(Base.Part.VAL),
        )

    def get_test_generator(self) -> Generator[tuple[np.ndarray, np.ndarray, int]]:
        """
        Return generator for test set.

        This generator yields batches of data suitable for testing.

        Returns:
            A generator yielding test data batches.
        """
        return self._get_batch_generator(Base.Part.TEST)

    def _compute_local_workload(self, nsamples: int) -> tuple[int, int, int]:
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
        _nsamples: float = nsamples

        if self.model.dataset_percentage != 0:
            _nsamples = nsamples * self.model.dataset_percentage

        if batches_per_worker > self.model.steps_per_epoch > 0:
            batches_per_worker = self.model.steps_per_epoch
            _nsamples = batches_per_worker * global_batch_size

        # Calculate nsamples per worker
        nsamples_per_worker, big_workers = divmod(_nsamples, self.model.nprocs)
        nsamples_per_big_worker = nsamples_per_worker + 1

        # Calculate local values
        if self.model.rank < big_workers:
            local_nsamples = nsamples_per_big_worker
            local_offset = self.model.rank * nsamples_per_big_worker
        else:
            local_nsamples = nsamples_per_worker
            local_offset = nsamples_per_big_worker * big_workers + nsamples_per_worker * (
                self.model.rank - big_workers
            )

        return int(local_offset), int(local_nsamples), int(_nsamples)

    def _model_init(self) -> None:
        """Generates initial self._x[] and self._y[]. To be implemented in derived classes."""
        self.x_empty_batch = np.zeros(
            shape=self.model.encode_shape((0, *self.input_shape)), dtype=self.model.dtype
        )
        self.y_empty_batch = np.zeros(shape=(0, *self.output_shape), dtype=self.model.dtype)

        # Declare _x and _y for train, val and test dataset parts
        self._x = [self.x_empty_batch] * len(Base.Part)
        self._y = [self.y_empty_batch] * len(Base.Part)

    def _data_generator(self, part: Base.Part) -> Generator[tuple[np.ndarray, np.ndarray]]:
        """Yield raw data from the dataset partition."""
        yield self._x[part], self._y[part]

    @staticmethod
    def _x_augment_adaptor(func: Callable[[np.ndarray], np.ndarray]) -> TransformFunc:
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

    def _augment_data_generator(self, part: Base.Part) -> Generator[tuple[np.ndarray, np.ndarray]]:
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

    def _batch_generator(self, part: Base.Part) -> Generator[tuple[np.ndarray, np.ndarray, int]]:
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
        # However to avoid communications per batch, we assume all process have
        # our x_local_batch.shape[0]

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

        generator = self._augment_data_generator(part)
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

            while (x_data.shape[0] > 0) and (
                (x_data.shape[0] >= local_batch_size) or (global_batch_size >= nsamples)
            ):
                x_batch, x_data = x_data[:local_batch_size], x_data[local_batch_size:]
                y_batch, y_data = y_data[:local_batch_size], y_data[local_batch_size:]

                global_batch_size = min(nsamples, global_batch_size)
                yield x_batch[:nsamples], y_batch[:nsamples], global_batch_size
                nsamples -= global_batch_size

    def _get_batch_generator(
        self, part: Base.Part
    ) -> Generator[tuple[np.ndarray, np.ndarray, int]]:
        """
        Yield batches with background prefetching.

        This method wraps the actual batch generator with a `BackgroundGenerator`
        to enable prefetching of batches, improving data loading performance.

        Args:
            part: The dataset partition (TRAIN, VAL, or TEST) to generate batches from.

        Yields:
            Tuples of (x_batch, y_batch, effective_global_batch_size), prefetched.
        """
        yield from BackgroundGenerator(self._batch_generator(part), max_prefetch=1)

        # NOTE: The following infinite loop provides of empty batches
        #       if there are asked more batches than actually are.
        x_empty_batch = np.zeros(
            shape=self.model.encode_shape((0, *self.input_shape)), dtype=self.model.dtype
        )
        y_empty_batch = np.zeros(shape=(0, *self.output_shape), dtype=self.model.dtype)
        while True:
            yield x_empty_batch, y_empty_batch, 0
