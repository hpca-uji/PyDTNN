"""
Dataset module for PyDTNN.

Provides the base Dataset class and utility functions for managing,
transforming, and generating data batches for machine learning models.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageEnhance
from scipy.ndimage import gaussian_filter

from pydtnn.datasets.abstract.base import Base
from pydtnn.datasets.abstract.init import Init
from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.tensor import TensorFormat

__all__ = ("Augment",)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pydtnn.model.utils import Utils as Model


type TransformFunc = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


class Augment(Init):
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
        super().__init__(
            model,
            train_nsamples,
            test_nsamples,
            input_shape,
            output_shape,
            force_test_as_validation,
            debug,
        )

        self._transforms = dict[Base.Part, list[TransformFunc]]()
        transforms_training = list[TransformFunc]()
        transforms_always = list[TransformFunc]()

        if self.model.input_crop:
            size = self.input_shape[1:]
            assert len(size) == 2
            crop, size = self._calculate_crop(size)
            self.input_shape = (self.input_shape[0], *size)
            transforms_training.append(self._x_transform_adaptor(self._transform_crop))
            transforms_always.append(self._x_transform_adaptor(self._transform_crop))

        if self.model.input_scale:
            self.input_shape = (
                self.input_shape[0],
                self.model.input_scale_size,
                self.model.input_scale_size,
            )
            transforms_training.append(self._x_transform_adaptor(self._transform_scale))
            transforms_always.append(self._x_transform_adaptor(self._transform_scale))

        if self.model.augment_horizontal_flip > 0:
            transforms_training.append(self._x_transform_adaptor(self._transform_horizontal_flip))

        if self.model.augment_vertical_flip > 0:
            transforms_training.append(self._x_transform_adaptor(self._transform_vertical_flip))

        if self.model.augment_brightness > 0:
            transforms_training.append(self._x_transform_adaptor(self._transform_brightness))

        if self.model.augment_contrast > 0:
            transforms_training.append(self._x_transform_adaptor(self._transform_contrast))

        if self.model.augment_saturation > 0:
            transforms_training.append(self._x_transform_adaptor(self._transform_saturation))

        if self.model.augment_blur > 0:
            transforms_training.append(self._x_transform_adaptor(self._transform_blur))

        if self.model.augment_mask > 0:
            transforms_training.append(self._x_transform_adaptor(self._transform_mask))

        if self.model.augment_perspective > 0:
            transforms_training.append(self._x_transform_adaptor(self._transform_perspective))

        if self.model.augment_rotate > 0:
            transforms_training.append(self._x_transform_adaptor(self._transform_rotate))

        if self.model.input_normalize:
            transforms_training.append(self._x_transform_adaptor(self._transform_normalize))
            transforms_always.append(self._x_transform_adaptor(self._transform_normalize))

        if self.model.augment_shuffle:
            transforms_training.append(self._augment_shuffle)
            transforms_always.append(self._augment_shuffle)

        self._transforms[Base.Part.TRAIN] = transforms_training
        self._transforms[Base.Part.TEST] = transforms_always
        self._transforms[Base.Part.VAL] = transforms_always

    @staticmethod
    def _x_transform_adaptor(func: Callable[[np.ndarray], np.ndarray]) -> TransformFunc:
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

    def _transform_data_generator(
        self, part: Base.Part
    ) -> Generator[tuple[np.ndarray, np.ndarray]]:
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
            transforms = self._transforms[part]
            for transform in transforms:
                x, y = transform(x, y)
            yield x, y

    def _transform_normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data using model parameters.

        Applies offset and scaling defined in `self.model.augment_normalize_offset`
        and `self.model.augment_normalize_scale` to the input data.

        Args:
            data: The input numpy array to normalize.

        Returns:
            The normalized numpy array.
        """
        if self.model.input_normalize_scale:
            scale = self.model.input_normalize_scale
            offset = self.model.input_normalize_offset
        else:
            scale = self.normal_scale
            offset = self.normal_offset
        np.add(data, offset, out=data)
        np.multiply(data, scale, out=data)
        return data

    def _transform_flip(
        self, data: np.ndarray, augment_probability: float, axis: int
    ) -> np.ndarray:
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

        s = np.where(self.model.random.random(n) <= augment_probability)[0]
        data[s, ...] = np.flip(data[s, ...], axis=axis)

        return data

    def _transform_horizontal_flip(self, data: np.ndarray) -> np.ndarray:
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
                raise NotImplementedError(
                    f"Dataset _do_augment_horizontal_flip is not implemented for {
                        self.model.tensor_format
                    } format."
                )

        return self._transform_flip(
            data=data, augment_probability=self.model.augment_horizontal_flip, axis=width_dim
        )

    def _transform_vertical_flip(self, data: np.ndarray) -> np.ndarray:
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
                raise NotImplementedError(
                    f"Dataset _do_augment_vertical_flip is not implemented for {
                        self.model.tensor_format
                    } format."
                )

        return self._transform_flip(
            data=data, augment_probability=self.model.augment_vertical_flip, axis=height_dim
        )

    def _augment_shuffle(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
        self.model.random.shuffle(idx)
        x[:] = x[idx]
        y[:] = y[idx]
        return x, y

    def _transform_mask(self, data: np.ndarray) -> np.ndarray:
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

        s = np.where(self.model.random.random(n) <= self.model.augment_mask)[0]

        t = self.model.random.integers(0, h - mask_size, (len(s),))
        ll = self.model.random.integers(0, w - mask_size, (len(s),))
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
                    raise NotImplementedError(
                        f"Dataset _do_augment_mask is not implemented for {
                            self.model.tensor_format
                        } format."
                    )
            data[ri, ...] = np.roll(
                data[ri, ...], self.model.random.integers(-t[i], (h - b)), axis=1
            )
            data[ri, ...] = np.roll(
                data[ri, ...], self.model.random.integers(-ll[i], (w - r)), axis=2
            )
        return data

    def _transform_rotate(self, data: np.ndarray) -> np.ndarray:
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
        _n, _c, _h, _w = data.shape
        # NOTE: _c not included so all channels in a sample rotate by the same amount
        rotation = (self.model.random.random(_n) - 0.5) * (2 * self.model.augment_rotate_degree)

        s = np.where(self.model.random.random(_n) <= self.model.augment_rotate)[0]

        for n in s:
            for c in range(_c):
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

    def _transform_blur(self, data: np.ndarray) -> np.ndarray:
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
        _n, _c, _h, _w = data.shape

        s = np.where(self.model.random.random(_n) <= self.model.augment_blur)[0]

        for n in s:
            data[n] = gaussian_filter(
                data[n], sigma=(0, self.model.augment_blur_size, self.model.augment_blur_size)
            )

        data = self.model.encode_tensor(data)

        return data

    def _transform_scale(self, data: np.ndarray) -> np.ndarray:
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

        size = (self.model.input_scale_size, self.model.input_scale_size)
        shape = (*data.shape[:2], *size)
        _n, _c, _h, _w = shape

        new_data = np.empty(shape=shape, dtype=self.model.dtype)

        for n in range(_n):
            for c in range(_c):
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

    def _calculate_crop(
        self, size: tuple[int, int]
    ) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
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
        frame_fraction = (1 - self.model.input_crop_perc) / 2
        x_offset, y_offset = round(width * frame_fraction), round(height * frame_fraction)
        crop = (x_offset, y_offset, width - x_offset, height - y_offset)
        size = (crop[2] - crop[0], crop[3] - crop[1])
        return (crop, size)

    def _transform_crop(self, data: np.ndarray) -> np.ndarray:
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
        _n, _c, _h, _w = shape

        new_data = np.empty(shape=shape, dtype=self.model.dtype)

        for n in range(_n):
            for c in range(_c):
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

    def _transform_brightness(self, data: np.ndarray) -> np.ndarray:
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
        _n, _c, _h, _w = data.shape
        # NOTE: _c not included so all channels in a sample rotate by the same amount
        brightness: np.ndarray = self.model.random.random(_n) * self.model.augment_brightness_factor

        s = np.where(self.model.random.random(_n) <= self.model.augment_brightness)[0]

        for n in s:
            for c in range(_c):
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

    def _transform_contrast(self, data: np.ndarray) -> np.ndarray:
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
        _n, _c, _h, _w = data.shape
        # NOTE: _c not included so all channels in a sample rotate by the same amount
        contrast: np.ndarray = self.model.random.random(_n) * self.model.augment_contrast_factor

        s = np.where(self.model.random.random(_n) <= self.model.augment_contrast)[0]

        for n in s:
            for c in range(_c):
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

    def _transform_saturation(self, data: np.ndarray) -> np.ndarray:
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
        _n, _c, _h, _w = data.shape
        # NOTE: _c not included so all channels in a sample rotate by the same amount
        saturation: np.ndarray = self.model.random.random(_n) * self.model.augment_saturation_factor

        s = np.where(self.model.random.random(_n) <= self.model.augment_saturation)[0]

        for n in s:
            for c in range(_c):
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

    def _transform_perspective(self, data: np.ndarray) -> np.ndarray:
        """
        Apply random perspective augmentation to images.

        Args:
            data: The input numpy array (batch of images).

        Returns:
            The array of images with some perspective change.
        """
        data = self.model.decode_tensor(data)
        _n, _c, _h, _w = data.shape
        # NOTE: _c not included so all channels in a sample rotate by the same amount
        persepctive: np.ndarray = (
            self.model.random.random(_n) * self.model.augment_perspective_factor
        )

        s = np.where(self.model.random.random(_n) <= self.model.augment_perspective)[0]

        for n in s:
            for c in range(_c):
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
    def _perspective_coeffs(
        src_points: np.ndarray | list, dst_points: np.ndarray | list
    ) -> list[int]:
        """
        Calculate perspective transformation coefficients.

        Args:
            src_points: Source points for the transformation.
            dst_points: Destination points for the transformation.

        Returns:
            A numpy array containing the transformation coefficients.
        """
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
        return np.array(res).reshape(8).tolist()

    def _image_perspective(self, image: Image.Image, factor: float) -> Image.Image:
        """
        Apply perspective transformation to a PIL image.

        Args:
            image: The input PIL image.
            factor: The intensity factor of the perspective transformation.

        Returns:
            The transformed PIL image.
        """
        # NOTE:
        # top_left     = [0, 0]
        # top_right    = [width, 0]
        # bottom_left  = [0, height]
        # bottom_right = [width, height]
        width, height = image.size

        top_left = (self.model.random.uniform(0, factor), self.model.random.uniform(0, factor))
        top_right = (1 - self.model.random.uniform(0, factor), self.model.random.uniform(0, factor))
        bottom_left = (
            self.model.random.uniform(0, factor),
            1 - self.model.random.uniform(0, factor),
        )
        bottom_right = (
            1 - self.model.random.uniform(0, factor),
            1 - self.model.random.uniform(0, factor),
        )

        transformed_points = list(zip(*[top_left, top_right, bottom_left, bottom_right]))
        widths = transformed_points[0]
        heights = transformed_points[1]

        diff_widths = max(widths) - min(widths)
        diff_heights = max(heights) - min(heights)

        if diff_widths < diff_heights:
            _min = min(heights)
            _max = max(heights)
        else:
            _min = min(widths)
            _max = max(widths)

        widths = np.interp(widths, (_min, _max), (0, 1))
        heights = np.interp(heights, (_min, _max), (0, 1))
        np.multiply(widths, width, out=widths)
        np.multiply(heights, height, out=heights)
        # NOTE: The change of dtype is made in two steps in order to store the data without truncate values to early
        #  Example:
        #   0.5 * 7.1 = 3,55 =(truncate)=> 3
        #   !=
        #   0.5 * 7.1 =(truncate variables)=> 0 * 7 = 0
        widths = np.asanyarray(widths, dtype=np.int32)
        heights = np.asanyarray(heights, dtype=np.int32)

        rescaled_base = np.asarray(
            [
                (width * _min, height * _min),
                (width * _max, height * _min),
                (width * _min, height * _max),
                (width * _max, height * _max),
            ],
            np.int32,
        )
        transformed_points = list(zip(widths, heights))

        coeffs = self._perspective_coeffs(transformed_points, rescaled_base)
        transomed_img = image.transform(
            (width, height),
            Image.Transform.PERSPECTIVE,
            coeffs,
            Image.Resampling.BICUBIC,
        )
        return transomed_img
