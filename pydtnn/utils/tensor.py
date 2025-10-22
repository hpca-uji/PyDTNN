from __future__ import annotations

from typing import assert_never
from enum import auto, StrEnum


class ChannelFormat(StrEnum):
    WH = auto()
    HW = auto()


class SampleFormat(StrEnum):
    WHC = auto()
    HWC = auto()
    CHW = auto()

    def as_channel(self) -> ChannelFormat:
        return ChannelFormat(self.strip("c"))

    def as_tensor(self) -> TensorFormat:
        return TensorFormat(f"n{self}")


class TensorFormat(StrEnum):
    NHWC = auto()
    NCHW = auto()

    def as_sample(self) -> SampleFormat:
        return SampleFormat(self.strip("n"))


def adjust_tensor_shape(shape: tuple[int, int, int, int], src: TensorFormat, dst: TensorFormat) -> tuple[int, int, int, int]:
    match src:
        case TensorFormat.NHWC:
            n, h, w, c = shape
        case TensorFormat.NCHW:
            n, c, h, w = shape
        case _:
            assert_never(src)

    match dst:
        case TensorFormat.NCHW:
            return (n, c, h, w)
        case TensorFormat.NHWC:
            return (n, h, w, c)
        case _:
            assert_never(dst)


def adjust_sample_shape(shape: tuple[int, int, int], src: SampleFormat, dst: SampleFormat) -> tuple[int, int, int]:
    match src:
        case SampleFormat.HWC:
            h, w, c = shape
        case SampleFormat.CHW:
            c, h, w = shape
        case SampleFormat.WHC:
            w, h, c = shape
        case _:
            assert_never(src)

    match dst:
        case SampleFormat.CHW:
            return (c, h, w)
        case SampleFormat.HWC:
            return (h, w, c)
        case SampleFormat.WHC:
            return (w, h, c)
        case _:
            assert_never(dst)


def adjust_channel_shape(shape: tuple[int, int], src: ChannelFormat, dst: ChannelFormat) -> tuple[int, int]:
    match src:
        case ChannelFormat.HW:
            h, w = shape
        case ChannelFormat.WH:
            w, h = shape
        case _:
            assert_never(src)

    match dst:
        case ChannelFormat.HW:
            return (h, w)
        case ChannelFormat.WH:
            return (w, h)
        case _:
            assert_never(dst)


def encode_tensor(shape, tensor_format=TensorFormat.NHWC):
    if len(shape) == 3 and tensor_format is TensorFormat.NCHW:
        return shape[2], shape[0], shape[1]
    else:  # Assuming PYDTNN_TENSOR_FORMAT.NHWC
        return shape


def decode_tensor(shape, tensor_format=TensorFormat.NHWC):
    if len(shape) == 3 and tensor_format is TensorFormat.NCHW:
        return shape[1], shape[2], shape[0]
    else:  # Assuming PYDTNN_TENSOR_FORMAT.NHWC
        return shape