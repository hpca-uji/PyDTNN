from __future__ import annotations

from typing import assert_never
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

    def reshape(self, shape: tuple[int, int], dst_order: ChannelFormat) -> tuple[int, int]:
        """
        Transform the `shape` from its current order to `dst_order` channel order.

        Args:
            shape (tuple[int, int, int, int]): source shape.
            dst_order (ChannelFormat): desired channel format.
        Returns:
            (tuple[int, int, int, int]): `shape` with `dst_order` channel order.
        """
        assert len(shape) == len(self), f"Unexpected dimensions (got: {shape}, expect: {self})"

        match self:
            case ChannelFormat.HW:
                h, w = shape
            case ChannelFormat.WH:
                w, h = shape
            case _:
                assert_never(self)

        match dst_order:
            case ChannelFormat.HW:
                return (h, w)
            case ChannelFormat.WH:
                return (w, h)
            case _:
                assert_never(dst_order)

    def transpose(self, data: np.ndarray, dst_format: ChannelFormat) -> np.ndarray:
        """
        Transpose elements of `data` from its current format to `dst_format` channel format.

        Args:
            data (np.ndarray): numpy array to transpose to a new channel format.
            dst_format (ChannelFormat): data's new channel format.
        Returns:
            np.ndarray: `data` with `dst_format` as it channel format.
        """
        return data.transpose(self.reshape(range(len(data.shape)), dst_format))  # type: ignore


class SampleFormat(StrEnum):
    CHW = auto()
    CWH = auto()
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

    def reshape(self, shape: tuple[int, int, int], dst_order: SampleFormat) -> tuple[int, int, int]:
        """
        Transform the `shape` from its current order to `dst_order` sample order.

        Args:
            shape (tuple[int, int, int, int]): source shape.
            dst_order (SampleFormat): desired sample format.
        Returns:
            (tuple[int, int, int, int]): `shape` with `dst_order` sample order.
        """
        assert len(shape) == len(self), f"Unexpected dimensions (got: {shape}, expect: {self})"

        match self:
            case SampleFormat.CHW:
                c, h, w = shape
            case SampleFormat.CWH:
                c, w, h = shape
            case SampleFormat.HWC:
                h, w, c = shape
            case SampleFormat.WHC:
                w, h, c = shape
            case _:
                assert_never(self)

        match dst_order:
            case SampleFormat.CHW:
                return (c, h, w)
            case SampleFormat.CWH:
                return (c, w, h)
            case SampleFormat.HWC:
                return (h, w, c)
            case SampleFormat.WHC:
                return (w, h, c)
            case _:
                assert_never(dst_order)

    def transpose(self, data: np.ndarray, dst_format: SampleFormat) -> np.ndarray:
        """
        Transpose elements of `data` from its current format to `dst_format` sample format.

        Args:
            data (np.ndarray): numpy array to transpose to a new sample format.
            dst_format (SampleFormat): data's new sample format.
        Returns:
            np.ndarray: `data` with `dst_format` as it sample format.
        """
        return data.transpose(self.reshape(range(len(data.shape)), dst_format))  # type: ignore


class TensorFormat(StrEnum):
    NCHW = auto()
    NCWH = auto()
    NHWC = auto()
    NWHC = auto()
    CHWN = auto()
    CWHN = auto()
    HWCN = auto()
    WHCN = auto()

    def as_sample(self) -> SampleFormat:
        """Down-cast format to just the sample"""
        return SampleFormat(self.strip("n"))

    def reshape(self, shape: tuple[int, int, int, int], dst_order: TensorFormat) -> tuple[int, int, int, int]:
        """
        Transform the `shape` from its current order to `dst_order` tensor order.

        Args:
            shape (tuple[int, int, int, int]): source shape.
            dst_order (TensorFormat): desired tensor format.
        Returns:
            (tuple[int, int, int, int]): `shape` with `dst_order` tensor order.
        """
        assert len(shape) == len(self), f"Unexpected dimensions (got: {shape}, expect: {self})"

        match self:
            case TensorFormat.NCHW:
                n, c, h, w = shape
            case TensorFormat.NCWH:
                n, c, w, h = shape
            case TensorFormat.NHWC:
                n, h, w, c = shape
            case TensorFormat.NWHC:
                n, w, h, c = shape
            case TensorFormat.CHWN:
                c, h, w, n = shape
            case TensorFormat.CWHN:
                c, w, h, n = shape
            case TensorFormat.HWCN:
                h, w, c, n = shape
            case TensorFormat.WHCN:
                w, h, c, n = shape
            case _:
                assert_never(self)

        match dst_order:
            case TensorFormat.NCHW:
                return (n, c, h, w)
            case TensorFormat.NCWH:
                return (n, c, w, h)
            case TensorFormat.NHWC:
                return (n, h, w, c)
            case TensorFormat.NWHC:
                return (n, w, h, c)
            case TensorFormat.CHWN:
                return (c, h, w, n)
            case TensorFormat.CWHN:
                return (c, w, h, n)
            case TensorFormat.HWCN:
                return (h, w, c, n)
            case TensorFormat.WHCN:
                return (w, h, c, n)
            case _:
                assert_never(dst_order)

    def transpose(self, data: np.ndarray, dst_format: TensorFormat) -> np.ndarray:
        """
        Transpose elements of `data` from its current format to `dst_format` tensor format.

        Args:
            data (np.ndarray): numpy array to transpose to a new tensor format.
            dst_format (TensorFormat): data's new tensor format.
        Returns:
            np.ndarray: `data` with `dst_format` as it tensor format.
        """
        return data.transpose(self.reshape(range(len(data.shape)), dst_format))  # type: ignore


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
