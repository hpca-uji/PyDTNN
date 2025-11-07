from __future__ import annotations

from typing import assert_never
from enum import auto, StrEnum

import numpy as np
from pydtnn.utils.types import ArrayShape


class ChannelFormat(StrEnum):
    WH = auto()
    HW = auto()

    def reshape(self, shape: tuple[int, int], format: ChannelFormat) -> tuple[int, int]:
        assert len(shape) == len(self), f"Unexpected dimensions (got: {shape}, expect: {self})"

        match self:
            case ChannelFormat.HW:
                h, w = shape
            case ChannelFormat.WH:
                w, h = shape
            case _:
                assert_never(self)

        match format:
            case ChannelFormat.HW:
                return (h, w)
            case ChannelFormat.WH:
                return (w, h)
            case _:
                assert_never(format)

    def transpose(self, data: np.ndarray, format: ChannelFormat) -> np.ndarray:
        return data.transpose(self.reshape(range(len(data.shape)), format))  # type: ignore


class SampleFormat(StrEnum):
    WHC = auto()
    HWC = auto()
    CHW = auto()

    def as_channel(self) -> ChannelFormat:
        return ChannelFormat(self.strip("c"))

    def as_tensor(self) -> TensorFormat:
        return TensorFormat(f"n{self}")

    def reshape(self, shape: tuple[int, int, int], format: SampleFormat) -> tuple[int, int, int]:
        assert len(shape) == len(self), f"Unexpected dimensions (got: {shape}, expect: {self})"

        match self:
            case SampleFormat.HWC:
                h, w, c = shape
            case SampleFormat.CHW:
                c, h, w = shape
            case SampleFormat.WHC:
                w, h, c = shape
            case _:
                assert_never(self)

        match format:
            case SampleFormat.CHW:
                return (c, h, w)
            case SampleFormat.HWC:
                return (h, w, c)
            case SampleFormat.WHC:
                return (w, h, c)
            case _:
                assert_never(format)

    def transpose(self, data: np.ndarray, format: SampleFormat) -> np.ndarray:
        return data.transpose(self.reshape(range(len(data.shape)), format))  # type: ignore


class TensorFormat(StrEnum):
    NHWC = auto()
    NCHW = auto()

    def as_sample(self) -> SampleFormat:
        return SampleFormat(self.strip("n"))

    def reshape(self, shape: tuple[int, int, int, int], dst: TensorFormat) -> tuple[int, int, int, int]:
        """
        Reshape \"shape"

        Args:
            shape (tuple[int, int, int, int]): New shape.
            dst (TensorFormat): New tensor format.
        Returns:
            (tuple[int, int, int, int]): \"shape\" with \"dst\" tensor format.
        """
        assert len(shape) == len(self), f"Unexpected dimensions (got: {shape}, expect: {self})"

        match self:
            case TensorFormat.NHWC:
                n, h, w, c = shape
            case TensorFormat.NCHW:
                n, c, h, w = shape
            case _:
                assert_never(self)

        match dst:
            case TensorFormat.NCHW:
                return (n, c, h, w)
            case TensorFormat.NHWC:
                return (n, h, w, c)
            case _:
                assert_never(dst)

    def transpose(self, data: np.ndarray, dst: TensorFormat) -> np.ndarray:
        """
        Transpose elements of \"data\" some it has the \"dst\" tensor format.

        Args:
            data (np.ndarray): numpy array to transpose to a new tensor format.
            dst (TensorFormat): The new data's tensor format.
        Returns:
            np.ndarray: \"data\" with \"dst\" tensor format.
        """
        return data.transpose(self.reshape(range(len(data.shape)), dst))  # type: ignore


def encode_tensor(shape: ArrayShape, dst_format=TensorFormat.NHWC) -> ArrayShape:
    """
    Returns the \"shape\" (exepcted in \"NHWC\") in \"dst_format\" format.
    Args:
        shape (ArrayShape): shape in \"HWC\" format.
        dst_format (TensorFormat): The new shape's tensor format. Default: \"TensorFormat.NHWC\".
    Returns:
        np.ndarray: \"shape\" with \"dst_format\" tensor format.
    """
    return SampleFormat.HWC.reshape(shape, dst_format.as_sample()) if len(shape) == 3 else shape


def decode_tensor(shape: ArrayShape, base_format=TensorFormat.NHWC) -> ArrayShape:
    """
    Returns the \"shape\" (exepcted in \"base_format\") in \"HWC\" format.
    Args:
        shape (ArrayShape): shape in \"base_format\" format.
        dst_format (TensorFormat): The base format. Default: \"TensorFormat.NHWC\".
    Returns:
        np.ndarray: \"shape\" with \"dst_format\" tensor format.
    """
    return base_format.as_sample().reshape(shape, SampleFormat.HWC) if len(shape) == 3 else shape
