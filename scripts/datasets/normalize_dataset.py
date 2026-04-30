#!/usr/bin/env python3

"""
Exports a PyDTNN dataset.
"""

import itertools
from collections import abc

import numpy as np

from pydtnn.model import Model
from pydtnn.datasets.dataset import Dataset
from pydtnn.utils.parser import PydtnnArgumentParser
from pydtnn.utils.tensor import TensorFormat


def compute_stats(iterator: abc.Iterable[np.ndarray]) -> tuple[float, float]:
    """Compute mean and standard diviation of batches"""
    n_total = 0
    mean = 0.0
    M2 = 0.0

    for batch in iterator:
        batch_size = batch.size

        batch_mean = batch.mean()
        batch_M2 = ((batch - batch_mean) ** 2).sum()

        if n_total == 0:
            mean = batch_mean
            M2 = batch_M2
            n_total = batch_size
        else:
            delta = batch_mean - mean
            total = n_total + batch_size

            mean += delta * batch_size / total
            M2 += batch_M2 + delta**2 * n_total * batch_size / total
            n_total = total

    variance = M2 / n_total
    std = np.sqrt(variance)
    return float(mean), float(std)


def get_full_x(dataset: Dataset) -> abc.Iterable[np.ndarray]:
    xy = itertools.chain(
        dataset._data_generator(Dataset.Part.TRAIN),
        dataset._data_generator(Dataset.Part.VAL),
        dataset._data_generator(Dataset.Part.TEST)
    )
    for x, y in xy:
        yield x


parser = PydtnnArgumentParser()
args = {**parser.to_dict(), "tensor_format": TensorFormat.NCHW, "batch_size": 1, "augment_shuffle": False}
model = Model(**args)

dataset = get_full_x(model.dataset)
mean, std = compute_stats(dataset)
offset, scale = -mean, 1.0 / std

print("Normalize:")
print(f"- offset: {offset:+.3f}")
print(f"- scale:  {scale:+.3f}")
