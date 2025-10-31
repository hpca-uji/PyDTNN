from __future__ import annotations

from typing import assert_never
from enum import auto, StrEnum

import numpy as np
from pydtnn.utils.types import ArrayShape


class ChannelFormat(StrEnum):
    WH = auto()
    HW = auto()

    def reshape(self, shape: tuple[int, int], format: ChannelFormat) -> tuple[int, int]:
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

    def reshape(self, shape: tuple[int, int, int, int], format: TensorFormat) -> tuple[int, int, int, int]:
        match self:
            case TensorFormat.NHWC:
                n, h, w, c = shape
            case TensorFormat.NCHW:
                n, c, h, w = shape
            case _:
                assert_never(self)

        match format:
            case TensorFormat.NCHW:
                return (n, c, h, w)
            case TensorFormat.NHWC:
                return (n, h, w, c)
            case _:
                assert_never(format)

    def transpose(self, data: np.ndarray, format: TensorFormat) -> np.ndarray:
        return data.transpose(self.reshape(range(len(data.shape)), format))  # type: ignore


def encode_tensor(shape: ArrayShape, tensor_format=TensorFormat.NHWC) -> ArrayShape:
    if len(shape) == 3 and tensor_format is TensorFormat.NCHW:
        return shape[2], shape[0], shape[1]
    else:  # Assuming PYDTNN_TENSOR_FORMAT.NHWC
        return shape


def decode_tensor(shape: ArrayShape, tensor_format=TensorFormat.NHWC) -> ArrayShape:
    if len(shape) == 3 and tensor_format is TensorFormat.NCHW:
        return shape[1], shape[2], shape[0]
    else:  # Assuming PYDTNN_TENSOR_FORMAT.NHWC
        return shape