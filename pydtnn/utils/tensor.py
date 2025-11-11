from __future__ import annotations

from enum import auto, StrEnum

import numpy as np
from pydtnn.utils.types import ArrayShape


class ChannelFormat(StrEnum):
    HW = auto()
    WH = auto()

    def as_sample(self) -> SampleFormat:
        """Up-cast format to include channel number (prefers left-side up-cast)"""
        try:
            return SampleFormat(f"c{self}")
        except ValueError:
            return SampleFormat(f"{self}c")

    def reshape(self, shape: tuple[int, int], dst: ChannelFormat) -> tuple[int, int]:
        """
        Transform the `shape` from its current order to `dst` channel order.

        Args:
            shape (tuple[int, int, int, int]): source shape.
            dst (ChannelFormat): desired channel format.
        Returns:
            (tuple[int, int, int, int]): `shape` with `dst` channel order.
        """
        return format_reshape(shape, self, dst)  # type: ignore

    def transpose(self, data: np.ndarray, dst: ChannelFormat) -> np.ndarray:
        """
        Transpose elements of `data` from its current format to `dst` channel format.

        Args:
            data (np.ndarray): numpy array to transpose to a new channel format.
            dst (ChannelFormat): data's new channel format.
        Returns:
            np.ndarray: `data` with `dst` as it channel format.
        """
        return format_transpose(data, self, dst)


class SampleFormat(StrEnum):
    CHW = auto()
    HWC = auto()
    CWH = auto()

    def as_channel(self) -> ChannelFormat:
        """Down-cast format to just the channel"""
        return ChannelFormat(self.strip("c"))

    def as_tensor(self) -> TensorFormat:
        """Up-cast format to include sample number (prefers left-side up-cast)"""
        try:
            return TensorFormat(f"n{self}")
        except ValueError:
            return TensorFormat(f"{self}n")

    def reshape(self, shape: tuple[int, int, int], dst: SampleFormat) -> tuple[int, int, int]:
        """
        Transform the `shape` from its current order to `dst` sample order.

        Args:
            shape (tuple[int, int, int, int]): source shape.
            dst (SampleFormat): desired sample format.
        Returns:
            (tuple[int, int, int, int]): `shape` with `dst` sample order.
        """
        return format_reshape(shape, self, dst)  # type: ignore

    def transpose(self, data: np.ndarray, dst: SampleFormat) -> np.ndarray:
        """
        Transpose elements of `data` from its current format to `dst` sample format.

        Args:
            data (np.ndarray): numpy array to transpose to a new sample format.
            dst (SampleFormat): data's new sample format.
        Returns:
            np.ndarray: `data` with `dst` as it sample format.
        """
        return format_transpose(data, self, dst)


class TensorFormat(StrEnum):
    NCHW = auto()
    NHWC = auto()

    def as_sample(self) -> SampleFormat:
        """Down-cast format to just the sample"""
        return SampleFormat(self.strip("n"))

    def reshape(self, shape: tuple[int, int, int, int], dst: TensorFormat) -> tuple[int, int, int, int]:
        """
        Transform the `shape` from its current order to `dst` tensor order.

        Args:
            shape (tuple[int, int, int, int]): source shape.
            dst (TensorFormat): desired tensor format.
        Returns:
            (tuple[int, int, int, int]): `shape` with `dst` tensor order.
        """
        return format_reshape(shape, self, dst)  # type: ignore

    def transpose(self, data: np.ndarray, dst: TensorFormat) -> np.ndarray:
        """
        Transpose elements of `data` from its current format to `dst` tensor format.

        Args:
            data (np.ndarray): numpy array to transpose to a new tensor format.
            dst (TensorFormat): data's new tensor format.
        Returns:
            np.ndarray: `data` with `dst` as it tensor format.
        """
        return format_transpose(data, self, dst)


def format_reshape(shape: tuple[int, ...], src: str, dst: str) -> tuple[int, ...]:
    """
    Transform the `shape` from its current `src` order to `dst` order.

    Args:
        shape (tuple[int, ...]): source shape data.
        src (str): current format.
        dst (str): desired format.
    Returns:
        (tuple[int, ...]): `shape` with `dst` order.
    """

    assert len(shape) == len(src) == len(dst), f"Inconsistent number of dimensions ({shape=}, {src=}, {dst=})"
    assert set(src) == set(dst), f"Inconsistent dimension names ({src=}, {dst=})"
    assert set(src) == len(set(src)), f"Duplicate dimension names ({src=})"
    assert set(dst) == len(set(dst)), f"Duplicate dimension names ({dst=})"

    dims = dict(zip(src, shape))
    return tuple(dims[i] for i in dst)  # type: ignore


def format_transpose(data: np.ndarray, src: str, dst: str) -> np.ndarray:
    """
    Transpose elements of `data` from its current `src` format to `dst` format.

    Args:
        data (np.ndarray): source numpy array.
        src (str): current format.
        dst (str): desired format.
    Returns:
        np.ndarray: `data` with `dst` format.
    """
    return data.transpose(format_reshape(range(len(data.shape)), src, dst))  # type: ignore


def encode_tensor(shape: ArrayShape, encoded_format=TensorFormat.NHWC) -> ArrayShape:
    """
    Returns the `shape` (exepcted in `HWC`) in `encoded_format` order.
    If `shape` does not have 3 dimensions, it is returned as-is.
    Args:
        shape (ArrayShape): shape in `HWC` format.
        encoded_format (TensorFormat): The encoded tensor format. Default: `TensorFormat.NHWC`.
    Returns:
        np.ndarray: `shape` with `encoded_format` tensor format.
    """
    return SampleFormat.HWC.reshape(shape, encoded_format.as_sample()) if len(shape) == 3 else shape


def decode_tensor(shape: ArrayShape, encoded_format=TensorFormat.NHWC) -> ArrayShape:
    """
    Returns the `shape` (exepcted in `encoded_format`) in `HWC` order.
    If `shape` does not have 3 dimensions, it is returned as-is.
    Args:
        shape (ArrayShape): shape in `encoded_format` format.
        encoded_format (TensorFormat): The encoded format. Default: `TensorFormat.NHWC`.
    Returns:
        np.ndarray: `shape` with `HWC` tensor format.
    """
    return encoded_format.as_sample().reshape(shape, SampleFormat.HWC) if len(shape) == 3 else shape
