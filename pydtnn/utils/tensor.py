from __future__ import annotations

from enum import auto, StrEnum

import numpy as np


# Formats
class ChannelFormat(StrEnum):
    HW = auto()
    WH = auto()

    def as_sample(self) -> SampleFormat:
        """Up-cast format to include channel number (prefers left-side up-cast)"""
        try:
            return SampleFormat(f"c{self}")
        except ValueError:
            return SampleFormat(f"{self}c")


class SampleFormat(StrEnum):
    CHW = auto()
    HWC = auto()
    WHC = auto()

    def as_channel(self) -> ChannelFormat:
        """Down-cast format to just the channel"""
        return ChannelFormat(self.strip("c"))

    def as_tensor(self) -> TensorFormat:
        """Up-cast format to include sample number (prefers left-side up-cast)"""
        try:
            return TensorFormat(f"n{self}")
        except ValueError:
            return TensorFormat(f"{self}n")


class TensorFormat(StrEnum):
    NCHW = auto()
    NHWC = auto()

    def as_sample(self) -> SampleFormat:
        """Down-cast format to just the sample"""
        return SampleFormat(self.strip("n"))


# Helpers
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
    assert len(src) == len(set(src)), f"Duplicate dimension names ({src=})"
    assert len(dst) == len(set(dst)), f"Duplicate dimension names ({dst=})"

    if src == dst:
        return shape
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
    if src == dst:
        return data
    return data.transpose(format_reshape(range(len(data.shape)), src, dst))  # type: ignore


# Encoders
def encode_shape(shape: tuple[int, ...], format: str = TensorFormat.NCHW) -> tuple[int, ...]:
    """
    Transform the `shape` from its current `NCHW` order to `format` order (or less).

    Args:
        shape (tuple[int, ...]): source shape data.
        format (str): encoded format (NCHW).
    Returns:
        (tuple[int, ...]): `shape` with encoded order.
    """
    return format_reshape(shape, TensorFormat.NCHW[-len(shape):], format[-len(shape):])


def decode_shape(shape: tuple[int, ...], format: str = TensorFormat.NCHW) -> tuple[int, ...]:
    """
    Transform the `shape` from its current `format` order to `NCHW` order (or less).

    Args:
        shape (tuple[int, ...]): encoded shape data.
        format (str): encoded format (NCHW).
    Returns:
        (tuple[int, ...]): `shape` with `NCHW` order.
    """
    return format_reshape(shape, format[-len(shape):], TensorFormat.NCHW[-len(shape):])


def encode_tensor(data: np.ndarray, format: str = TensorFormat.NCHW) -> np.ndarray:
    """
    Transpose elements of `data` from its current `NCHW` format to `format` format.

    Args:
        data (np.ndarray): source data.
        format (str): encoded format (NCHW).
    Returns:
        (np.ndarray): `data` with encoded order.
    """
    return format_transpose(data, TensorFormat.NCHW[-len(data.shape):], format[-len(data.shape):])


def decode_tensor(data: np.ndarray, format: str = TensorFormat.NCHW) -> np.ndarray:
    """
    Transpose elements of `data` from its current `format` format to `NCHW` format.

    Args:
        data (np.ndarray): encoded data.
        format (str): encoded format (NCHW).
    Returns:
        (np.ndarray): `data` with `NCHW` order.
    """
    return format_transpose(data, format[-len(data.shape):], TensorFormat.NCHW[-len(data.shape):])
