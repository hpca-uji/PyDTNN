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
from PIL import Image
from scipy.ndimage import gaussian_filter

from pydtnn.datasets.abstract.base import Base
from pydtnn.datasets.abstract.init import Init
from pydtnn.utils import random
from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.tensor import TensorFormat

__all__ = ("Transform",)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pydtnn.model.utils import Utils as Model


type TransformFunc = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


class Transform(Init):
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
        Initialize the Transform dataset handler.

        Args:
            model: The model utility instance containing configuration.
            train_nsamples: Number of training samples.
            test_nsamples: Number of test samples.
            input_shape: Expected shape of input data.
            output_shape: Expected shape of output data.
            force_test_as_validation: Whether to treat test set as validation.
            debug: Enable debug mode.
        """
        super().__init__(model, train_nsamples, test_nsamples, input_shape, output_shape, force_test_as_validation, debug)

        self._transformations = dict[Base.Part, list[TransformFunc]]()
        transformations_training = list[TransformFunc]()
        transformations_always = list[TransformFunc]()

        if self.model.transform_crop:
            crop, size = self._calculate_crop(self.input_shape[1:])  # type: ignore (The cropped input shape will be a tuple[int, int])
            self.input_shape = (self.input_shape[0], *size)
            transformations_training.append(self._x_transformer_adaptor(self._do_transform_crop))
            transformations_always.append(self._x_transformer_adaptor(self._do_transform_crop))

        if self.model.transform_resize:
            self.input_shape = (self.input_shape[0], self.model.transform_resize_size, self.model.transform_resize_size)
            transformations_training.append(self._x_transformer_adaptor(self._do_transform_resize))
            transformations_always.append(self._x_transformer_adaptor(self._do_transform_resize))

        if self.model.augment_blur > 0:
            transformations_training.append(self._x_transformer_adaptor(self._do_augment_blur))

        if self.model.augment_flip > 0:
            transformations_training.append(self._x_transformer_adaptor(self._do_augment_flip))

        if self.model.augment_mask > 0:
            transformations_training.append(self._x_transformer_adaptor(self._do_augment_mask))

        if self.model.augment_rotate > 0:
            transformations_training.append(self._x_transformer_adaptor(self._do_augment_rotate))

        if self.model.augment_shuffle:
            transformations_training.append(self._do_augment_shuffle)

        if self.model.normalize:
            transformations_training.append(self._x_transformer_adaptor(self._do_normalize))
            transformations_always.append(self._x_transformer_adaptor(self._do_normalize))

        self._transformations[Base.Part.TRAIN] = transformations_training
        self._transformations[Base.Part.TEST] = transformations_always
        self._transformations[Base.Part.VAL] = transformations_always

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
            for transformation in self._transformations[part]:
                x, y = transformation(x, y)
            yield x, y

    def _do_normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data using model parameters.

        Applies offset and scaling defined in `self.model.normalize_offset`
        and `self.model.normalize_scale` to the input data.

        Args:
            data: The input numpy array to normalize.

        Returns:
            The normalized numpy array.
        """
        np.add(data, self.model.normalize_offset, out=data)
        np.multiply(data, self.model.normalize_scale, out=data)
        return data

    def _do_augment_flip(self, data: np.ndarray) -> np.ndarray:
        """
        Apply random flip augmentation to images.

        Randomly flips a portion of the images based on the
        `self.model.augment_flip` parameter.

        Args:
            data: The input numpy array (batch of images).

        Returns:
            The array with some images potentially flipped.

        Raises:
            NotImplementedError: If the `self.model.tensor_format` is not supported.
        """
        n = data.shape[0]
        match self.model.tensor_format:
            case TensorFormat.NCHW:
                height_dim = 2
                width_dim = 3
            case TensorFormat.NHWC:
                height_dim = 1
                width_dim = 2
            case _:
                raise NotImplementedError(f"Dataset _do_augment_flip is not implemented for {self.model.tensor_format} format.")

        limit = min(n, int(n * self.model.augment_flip))

        s = np.arange(n)
        random.shuffle(s)
        s = s[:limit]
        data[s, ...] = np.flip(data[s, ...], axis=height_dim)

        s = np.arange(n)
        random.shuffle(s)
        s = s[:limit]
        data[s, ...] = np.flip(data[s, ...], axis=width_dim)

        return data

    def _do_augment_shuffle(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Shuffle samples within a batch.

        Randomly shuffles the order of samples (and their corresponding labels)
        within a given batch. This is a form of data augmentation.

        Args:
            x: The input data batch.
            y: The corresponding label batch.

        Returns:
            A tuple containing the shuffled input data and label batches.
        """
        idx = np.arange(x.shape[0])
        random.shuffle(idx)
        x[:] = x[idx]
        y[:] = y[idx]
        return x, y

    def _do_augment_mask(self, data: np.ndarray) -> np.ndarray:
        """
        Apply random mask augmentation.

        Randomly masks a portion of the images in the batch. The mask size and
        the percentage of images to mask are determined by model parameters.
        Pads the masked area with zeros.

        Args:
            data: The input numpy array (batch of images).

        Returns:
            The array with some images randomly masked.

        Raises:
            NotImplementedError: If the `self.model.tensor_format` is not supported.
        """
        n, c, h, w = self.model.decode_shape(data.shape)
        mask_size = min(self.model.augment_mask_size, h, w)
        limit = min(n, int(n * self.model.augment_mask))
        s = np.arange(n)
        random.shuffle(s)
        s = s[:limit]
        t = random.integers(0, h - mask_size, (limit,))
        ll = random.integers(0, w - mask_size, (limit,))
        for i, ri in enumerate(s):
            b, r = t[i] + mask_size, ll[i] + mask_size
            match self.model.tensor_format:
                case TensorFormat.NCHW:
                    data[ri, :, : t[i], : ll[i]] = 0.0
                    data[ri, :, b:, r:] = 0.0
                case TensorFormat.NHWC:
                    data[ri, : t[i], : ll[i], :] = 0.0
                    data[ri, b:, r:, :] = 0.0
                case _:
                    raise NotImplementedError(f"Dataset _do_augment_mask is not implemented for {self.model.tensor_format} format.")
            data[ri, ...] = np.roll(data[ri, ...], random.integers(-t[i], (h - b)), axis=1)
            data[ri, ...] = np.roll(data[ri, ...], random.integers(-ll[i], (w - r)), axis=2)
        return data

    def _do_augment_rotate(self, data: np.ndarray) -> np.ndarray:
        """
        Apply random rotation augmentation to images.

        Randomly rotate the images based on the
        `self.model.augment_rotate` parameter.

        Args:
            data: The input numpy array (batch of images).

        Returns:
            The array with some images potentially rotations.

        Raises:
            NotImplementedError: If the `self.model.tensor_format` is not supported.
        """
        data = self.model.decode_tensor(data)
        N, C, H, W = data.shape

        rotation = random.random(N) * 360

        limit = min(N, int(N * self.model.augment_mask))
        s = np.arange(N)
        random.shuffle(s)
        s = s[:limit]

        for n in s:
            for c in range(C):
                channel: np.ndarray = data[n, c]
                # NOTE: PIL mode F is WH in float32
                channel = channel.transpose().astype(np.float32)  # type: ignore (it's possible to use copy=None)
                image = Image.fromarray(channel, mode="F")
                image = image.rotate(rotation[n])
                channel = np.asarray(image, dtype=np.float32, order="C")
                channel = channel.transpose().astype(self.model.dtype)  # type: ignore (it's possible to use copy=None)
                data[n, c] = channel

        data = self.model.encode_tensor(data)

        return data

    def _do_augment_blur(self, data: np.ndarray) -> np.ndarray:
        """
        Apply random blur augmentation to images.

        Args:
            data: The input numpy array (batch of images).

        Returns:
            The array with some images potentially blurs.

        Raises:
            NotImplementedError: If the `self.model.tensor_format` is not supported.
        """
        data = self.model.decode_tensor(data)
        N, C, H, W = data.shape

        limit = min(N, int(N * self.model.augment_blur))
        s = np.arange(N)
        random.shuffle(s)
        s = s[:limit]

        for n in s:
            data[n] = gaussian_filter(data[n], sigma=(0, self.model.augment_blur_size, self.model.augment_blur_size))

        data = self.model.encode_tensor(data)

        return data

    def _do_transform_resize(self, data: np.ndarray) -> np.ndarray:
        """
        Resize images using PIL.

        Resizes images in the batch to a fixed size specified by
        `self.model.transform_resize_size`. Uses PIL for image resizing.

        Args:
            data: The input numpy array (batch of images).

        Returns:
            The array with images resized.
        """
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
        """
        Calculate crop coordinates and resulting size.

        Determines the bounding box for a center crop based on the
        `self.model.transform_crop_perc` parameter, and calculates the
        resulting dimensions after cropping.

        Args:
            size: The original (width, height) of the image.

        Returns:
            A tuple containing:
            - crop: A tuple (x1, y1, x2, y2) representing the crop box.
            - size: A tuple (new_width, new_height) representing the dimensions
                    after cropping.
        """
        width, height = size
        frame_fraction = (1 - self.model.transform_crop_perc) / 2
        x_offset, y_offset = round(width * frame_fraction), round(height * frame_fraction)
        crop = (x_offset, y_offset, width - x_offset, height - y_offset)
        size = (crop[2] - crop[0], crop[3] - crop[1])
        return (crop, size)

    def _do_transform_crop(self, data: np.ndarray) -> np.ndarray:
        """
        Apply center crop transformation.

        Performs a center crop on the images in the batch according to the
        calculated crop box and size. Uses PIL for the cropping operation.

        Args:
            data: The input numpy array (batch of images).

        Returns:
            The array with images center-cropped.
        """
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
