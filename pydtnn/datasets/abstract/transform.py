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
from PIL import Image, ImageEnhance
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

        self._augments = dict[Base.Part, list[TransformFunc]]()
        augments_training = list[TransformFunc]()
        augments_always = list[TransformFunc]()

        if self.model.augment_crop:
            crop, size = self._calculate_crop(self.input_shape[1:])  # type: ignore (The cropped input shape will be a tuple[int, int])
            self.input_shape = (self.input_shape[0], *size)
            augments_training.append(self._x_augment_adaptor(self._do_augment_crop))
            augments_always.append(self._x_augment_adaptor(self._do_augment_crop))

        if self.model.augment_scale:
            self.input_shape = (self.input_shape[0], self.model.augment_scale_size, self.model.augment_scale_size)
            augments_training.append(self._x_augment_adaptor(self._do_augment_scale))
            augments_always.append(self._x_augment_adaptor(self._do_augment_scale))

        if self.model.augment_horizontal_flip > 0:
            augments_training.append(self._x_augment_adaptor(self._do_augment_horizontal_flip))

        if self.model.augment_vertical_flip > 0:
            augments_training.append(self._x_augment_adaptor(self._do_augment_vertical_flip))

        if self.model.augment_brightness > 0:
            augments_training.append(self._x_augment_adaptor(self._do_augment_brightness))

        if self.model.augment_contrast > 0:
            augments_training.append(self._x_augment_adaptor(self._do_augment_contrast))

        if self.model.augment_saturation > 0:
            augments_training.append(self._x_augment_adaptor(self._do_augment_saturation))

        if self.model.augment_blur > 0:
            augments_training.append(self._x_augment_adaptor(self._do_augment_blur))

        if self.model.augment_mask > 0:
            augments_training.append(self._x_augment_adaptor(self._do_augment_mask))

        if self.model.augment_perspective > 0:
            augments_training.append(self._x_augment_adaptor(self._do_augment_perspective))

        if self.model.augment_rotate > 0:
            augments_training.append(self._x_augment_adaptor(self._do_augment_rotate))

        if self.model.augment_normalize:
            augments_training.append(self._x_augment_adaptor(self._do_augment_normalize))
            augments_always.append(self._x_augment_adaptor(self._do_augment_normalize))

        if self.model.augment_shuffle:
            augments_training.append(self._do_augment_shuffle)

        self._augments[Base.Part.TRAIN] = augments_training
        self._augments[Base.Part.TEST] = augments_always
        self._augments[Base.Part.VAL] = augments_always

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
            for transformation in self._augments[part]:
                x, y = transformation(x, y)
            yield x, y

    def _do_augment_normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data using model parameters.

        Applies offset and scaling defined in `self.model.augment_normalize_offset`
        and `self.model.augment_normalize_scale` to the input data.

        Args:
            data: The input numpy array to normalize.

        Returns:
            The normalized numpy array.
        """
        np.add(data, self.model.augment_normalize_offset, out=data)
        np.multiply(data, self.model.augment_normalize_scale, out=data)
        return data

    def _do_augment_flip(self, data: np.ndarray, augment_probability: float, axis: int) -> np.ndarray:
        """
        Apply random flip augmentation to images.

        Args:
            data: The input numpy array (batch of images).
            augment_probability: The probability to do the flip
            axis: The axis about which the flip is done.

        Returns:
            The array with some images potentially flipped.
        """
        n = data.shape[0]

        s = np.where(random.random(n) <= augment_probability)[0]
        data[s, ...] = np.flip(data[s, ...], axis=axis)

        return data

    def _do_augment_horizontal_flip(self, data: np.ndarray) -> np.ndarray:
        """
        Apply random horizontal flip augmentation to images.

        Randomly flips a portion of the images based on the
        `self.model.augment_horizontal_flip` parameter.

        Args:
            data: The input numpy array (batch of images).

        Returns:
            The array with some images potentially flipped.

        Raises:
            NotImplementedError: If the `self.model.tensor_format` is not supported.
        """
        match self.model.tensor_format:
            case TensorFormat.NCHW:
                width_dim = 3
            case TensorFormat.NHWC:
                width_dim = 2
            case _:
                raise NotImplementedError(f"Dataset _do_augment_horizontal_flip is not implemented for {self.model.tensor_format} format.")

        return self._do_augment_flip(data=data, augment_probability=self.model.augment_horizontal_flip, axis=width_dim)

    def _do_augment_vertical_flip(self, data: np.ndarray) -> np.ndarray:
        """
        Apply random vertical flip augmentation to images.

        Randomly flips a portion of the images based on the
        `self.model.augment_vertical_flip` parameter.

        Args:
            data: The input numpy array (batch of images).

        Returns:
            The array with some images potentially flipped.

        Raises:
            NotImplementedError: If the `self.model.tensor_format` is not supported.
        """
        match self.model.tensor_format:
            case TensorFormat.NCHW:
                height_dim = 2
            case TensorFormat.NHWC:
                height_dim = 1
            case _:
                raise NotImplementedError(f"Dataset _do_augment_vertical_flip is not implemented for {self.model.tensor_format} format.")

        return self._do_augment_flip(data=data, augment_probability=self.model.augment_vertical_flip, axis=height_dim)

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

        s = np.where(random.random(n) <= self.model.augment_mask)[0]

        t = random.integers(0, h - mask_size, (len(s),))
        ll = random.integers(0, w - mask_size, (len(s),))
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
        `self.model.augment_rotate` and `self.model.augment_rotate_degree` parameters.

        Args:
            data: The input numpy array (batch of images).

        Returns:
            The array with some images potentially rotations.

        Raises:
            NotImplementedError: If the `self.model.tensor_format` is not supported.
        """
        data = self.model.decode_tensor(data)
        N, C, H, W = data.shape
        # NOTE: C not included so all channels in a sample rotate by the same amount
        rotation = (random.random(N) - 0.5) * (2 * self.model.augment_rotate_degree)

        s = np.where(random.random(N) <= self.model.augment_rotate)[0]

        for n in s:
            for c in range(C):
                channel: np.ndarray = data[n, c]
                # NOTE: PIL mode F is WH in float32
                channel = channel.transpose().astype(np.float32)
                image = Image.fromarray(channel, mode="F")
                image = image.rotate(rotation[n])
                channel = np.asarray(image, dtype=np.float32)
                channel = channel.transpose().astype(self.model.dtype)
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

        s = np.where(random.random(N) <= self.model.augment_blur)[0]

        for n in s:
            data[n] = gaussian_filter(data[n], sigma=(0, self.model.augment_blur_size, self.model.augment_blur_size))

        data = self.model.encode_tensor(data)

        return data

    def _do_augment_scale(self, data: np.ndarray) -> np.ndarray:
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

        size = (self.model.augment_scale_size, self.model.augment_scale_size)
        shape = (*data.shape[:2], *size)
        N, C, H, W = shape

        new_data = np.empty(shape=shape, dtype=self.model.dtype)

        for n in range(N):
            for c in range(C):
                channel: np.ndarray = data[n, c]
                # NOTE: PIL mode F is WH in float32
                channel = channel.transpose().astype(np.float32)
                image = Image.fromarray(channel, mode="F")
                image = image.resize(size)
                channel = np.asarray(image, dtype=np.float32)
                channel = channel.transpose().astype(self.model.dtype)
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
        frame_fraction = (1 - self.model.augment_crop_perc) / 2
        x_offset, y_offset = round(width * frame_fraction), round(height * frame_fraction)
        crop = (x_offset, y_offset, width - x_offset, height - y_offset)
        size = (crop[2] - crop[0], crop[3] - crop[1])
        return (crop, size)

    def _do_augment_crop(self, data: np.ndarray) -> np.ndarray:
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
                channel = channel.transpose().astype(np.float32)
                image = Image.fromarray(channel, mode="F")
                image = image.crop(crop)
                channel = np.asarray(image, dtype=np.float32)
                channel = channel.transpose().astype(self.model.dtype)
                new_data[n, c] = channel

        new_data = self.model.encode_tensor(new_data)

        return new_data

    def _do_augment_brightness(self, data: np.ndarray) -> np.ndarray:
        """
        Apply random brightness augmentation to images.

        Randomly rotates the images based on the
        `self.model.augment_brightness` and `self.model.augment_brightness_range` parameter.

        Args:
            data: The input numpy array (batch of images).

        Returns:
            The array of images with some brightness change.

        Raises:
            NotImplementedError: If the `self.model.tensor_format` is not supported.
        """
        data = self.model.decode_tensor(data)
        N, C, H, W = data.shape
        # NOTE: C not included so all channels in a sample rotate by the same amount
        brightness: np.ndarray = random.random(N) * self.model.augment_brightness_factor

        s = np.where(random.random(N) <= self.model.augment_brightness)[0]

        for n in s:
            for c in range(C):
                channel: np.ndarray = data[n, c]
                # NOTE: PIL mode F is WH in float32
                channel = np.interp(channel, (0, 1), (0, 255))
                channel = channel.transpose().astype(np.uint8)
                image = Image.fromarray(channel, mode="L")
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(brightness[n].item())
                channel = np.asarray(image, dtype=np.uint8)
                channel = channel.transpose().astype(self.model.dtype)
                channel = np.interp(channel, (0, 255), (0, 1))
                data[n, c] = channel

        data = self.model.encode_tensor(data)

        return data

    def _do_augment_contrast(self, data: np.ndarray) -> np.ndarray:
        """
        Apply random contrast augmentation to images.

        Randomly changes the contrast of the images based on the
        `self.model.augment_contrast` and `self.model.augment_contrast_range` parameter.

        Args:
            data: The input numpy array (batch of images).

        Returns:
            The array of images with some contrast change.

        Raises:
            NotImplementedError: If the `self.model.tensor_format` is not supported.
        """
        data = self.model.decode_tensor(data)
        N, C, H, W = data.shape
        # NOTE: C not included so all channels in a sample rotate by the same amount
        contrast: np.ndarray = random.random(N) * self.model.augment_contrast_factor

        s = np.where(random.random(N) <= self.model.augment_contrast)[0]

        for n in s:
            for c in range(C):
                channel: np.ndarray = data[n, c]
                channel = np.interp(channel, (0, 1), (0, 255))
                channel = channel.transpose().astype(np.uint8)
                image = Image.fromarray(channel, mode="L")
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(contrast[n].item())
                channel = np.asarray(image, dtype=np.uint8)
                channel = channel.transpose().astype(self.model.dtype)
                channel = np.interp(channel, (0, 255), (0, 1))
                data[n, c] = channel

        data = self.model.encode_tensor(data)

        return data

    def _do_augment_saturation(self, data: np.ndarray) -> np.ndarray:
        """
        Apply random saturation augmentation to images.

        Randomly changes the saturation the images based on the
        `self.model.augment_saturation` and `self.model.augment_saturation_range` parameter.

        Args:
            data: The input numpy array (batch of images).

        Returns:
            The array of images with some saturation change.

        Raises:
            NotImplementedError: If the `self.model.tensor_format` is not supported.
        """
        data = self.model.decode_tensor(data)
        N, C, H, W = data.shape
        # NOTE: C not included so all channels in a sample rotate by the same amount
        saturation: np.ndarray = random.random(N) * self.model.augment_saturation_factor

        s = np.where(random.random(N) <= self.model.augment_saturation)[0]

        for n in s:
            for c in range(C):
                channel: np.ndarray = data[n, c]
                channel = np.interp(channel, (0, 1), (0, 255))
                channel = channel.transpose().astype(np.uint8)
                image = Image.fromarray(channel, mode="L")
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(saturation[n].item())
                channel = np.asarray(image, dtype=np.uint8)
                channel = channel.transpose().astype(self.model.dtype)
                channel = np.interp(channel, (0, 255), (0, 1))
                data[n, c] = channel

        data = self.model.encode_tensor(data)

        return data

    def _do_augment_perspective(self, data: np.ndarray) -> np.ndarray:

        N, C, H, W = data.shape
        # NOTE: C not included so all channels in a sample rotate by the same amount
        persepctive: np.ndarray = random.random(N) * self.model.augment_perspective_factor

        s = np.where(random.random(N) <= self.model.augment_perspective)[0]

        for n in s:
            for c in range(C):
                channel: np.ndarray = data[n, c]
                channel = np.interp(channel, (0, 1), (0, 255))
                channel = channel.transpose().astype(np.uint8)
                image = Image.fromarray(channel, mode="L")
                image = self._image_perspective(image, persepctive[n].item())
                channel = np.asarray(image, dtype=np.uint8)
                channel = channel.transpose().astype(self.model.dtype)
                channel = np.interp(channel, (0, 255), (0, 1))
                data[n, c] = channel

        data = self.model.encode_tensor(data)

        return data

    @staticmethod
    def _perspective_coeffs(src_points: np.ndarray | list, dst_points: np.ndarray | list) -> np.ndarray[tuple[int]]:
        # Source:
        # A) https://stackoverflow.com/questions/14177744/how-does-perspective-transformation-work-in-pil
        # B) https://web.archive.org/web/20150222120106/xenia.media.mit.edu/~cwren/interpolator/
        matrix = []
        for p1, p2 in zip(src_points, dst_points):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

        matrix_a = np.matrix(matrix, dtype=np.float32)
        matrix_b = np.array(dst_points).reshape(8)

        res = np.dot(np.linalg.inv(matrix_a.T * matrix_a) * matrix_a.T, matrix_b)
        return np.array(res).reshape(8)

    def _image_perspective(self, image: Image.Image, factor: float) -> Image.Image:
        # NOTE:
        # top_left     = [0, 0]
        # top_right    = [width, 0]
        # bottom_left  = [0, height]
        # bottom_right = [width, height]
        width, height = image.size

        top_left = (random.uniform(0, factor), random.uniform(0, factor))
        top_right = (1 - random.uniform(0, factor), random.uniform(0, factor))
        bottom_left = (random.uniform(0, factor), 1 - random.uniform(0, factor))
        bottom_right = (1 - random.uniform(0, factor), 1 - random.uniform(0, factor))

        transformed_points = list(zip(*[top_left, top_right, bottom_left, bottom_right]))
        w = transformed_points[0]
        h = transformed_points[1]

        mw = max(w) - min(w)
        mh = max(h) - min(h)

        if mw < mh:
            _min = min(h)
            _max = max(h)
        else:
            _min = min(w)
            _max = max(w)

        w = np.interp(w, (_min, _max), (0, 1))
        h = np.interp(h, (_min, _max), (0, 1))
        np.multiply(w, width, out=w)
        np.multiply(h, height, out=h)
        w = np.asanyarray(w, dtype=np.int32)
        h = np.asanyarray(h, dtype=np.int32)

        rescaled_base = np.asarray([(width * _min, height * _min),
                                    (width * _max, height * _min),
                                    (width * _min, height * _max),
                                    (width * _max, height * _max)], np.int32)
        transformed_points = list(zip(w, h))

        coeffs = self._perspective_coeffs(transformed_points, rescaled_base)
        transomed_img = image.transform((width, height),
                                        Image.Transform.PERSPECTIVE,
                                        coeffs,  # type: ignore (It's the right type)
                                        Image.Resampling.BICUBIC)
        return transomed_img
