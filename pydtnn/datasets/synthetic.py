from typing import TYPE_CHECKING

import numpy as np

from pydtnn.datasets.dataset import Dataset
from pydtnn.utils.tensor import TensorFormat

if TYPE_CHECKING:
    from pydtnn.model import Model


class Synthetic(Dataset):
    """
    Synthetic Dataset

    Generates random data from:
    - `model.synthetic_train_samples`
    - `model.synthetic_test_samples`
    - `model.synthetic_input_shape` (coma separated)
    - `model.synthetic_output_shape` (coma separated)
    """

    def __init__(self, model: "Model", force_test_as_validation=False, debug=False):
        train_nsamples = int(model.synthetic_train_samples)
        test_nsamples = int(model.synthetic_test_samples)
        input_shape = tuple(map(int, model.synthetic_input_shape.split(",")))
        output_shape = tuple(map(int, model.synthetic_output_shape.split(",")))

        super().__init__(model,
                         train_nsamples=train_nsamples,
                         test_nsamples=test_nsamples,
                         input_shape=input_shape,
                         output_shape=output_shape,
                         force_test_as_validation=force_test_as_validation,
                         debug=debug)

    def _init_actual_data(self):
        self._x = [np.empty((0,)) for part in Dataset.Part]
        self._y = [np.empty((0,)) for part in Dataset.Part]

        for part in Dataset.Part:
            local_batches = self._local_nsamples[part] // self.model.batch_size
            nsamples = local_batches * self.model.batch_size
            x_shape = (nsamples, *self.input_shape)
            x_shape = self.model.encode_shape(x_shape)  # type: ignore
            y_shape = (nsamples, *self.output_shape)
            self._x[part] = np.zeros(x_shape, dtype=self.model.dtype, order="C")
            self._y[part] = np.zeros(y_shape, dtype=self.model.dtype, order="C")

    def _actual_data_generator(self, part: Dataset.Part):
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
        for p in Dataset.Part:
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
