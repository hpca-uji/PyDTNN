"""
State module for PyDTNN.

Provides the State class for managing dataset exports, splitting, and
archiving operations within the PyDTNN framework.
"""

import itertools
from pathlib import Path
from typing import Generator

import numpy as np

from pydtnn.datasets.abstract.base import Base
from pydtnn.datasets.abstract.init import Init
from pydtnn.utils import BackgroundGenerator

__all__ = ("State",)


class State(Init):
    """
    Handles dataset state management, including exporting and archiving.

    Extends the Init class to provide functionality for converting dataset
    generators into static numpy arrays and saving them to disk.
    """

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
        gen_train = BackgroundGenerator(self._batch_generator(Base.Part.TRAIN), max_prefetch=1)
        gen_val = BackgroundGenerator(self._batch_generator(Base.Part.VAL), max_prefetch=1)
        gen_test = BackgroundGenerator(self._batch_generator(Base.Part.TEST), max_prefetch=1)
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
            "name": self.name,  # pyright: ignore[reportAttributeAccessIssue]
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test,
        }

    def _export_split(
        self, data: dict[str, np.ndarray], split_weights: list[float] = [1.0]
    ) -> Generator[dict[str, np.ndarray]]:
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
            yield {
                **data,
                "x_train": x_train,
                "y_train": y_train,
                "x_test": x_test,
                "y_test": y_test,
            }

    def export_archive(
        self, path: Path | None = None, split_weights: list[float] | None = None
    ) -> None:
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
                np.savez_compressed(path / f"archive.{split}.npz", **data)  # pyright: ignore[reportArgumentType]
        else:
            np.savez_compressed(path / "archive.npz", **data)  # pyright: ignore[reportArgumentType]
