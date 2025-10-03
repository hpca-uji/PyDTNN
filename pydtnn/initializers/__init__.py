"""
PyDTNN initializers
"""

import numpy as np
import scipy.stats as stats
from enum import StrEnum, auto
from typing import Callable


class DistributionModeEnum(StrEnum):
    FAN_IN = auto()
    FAN_OUT = auto()
    FAN_AVG = auto()


class ProbabilisticDistributionEnum(StrEnum):
    UNIFORM = auto()
    NORMAL = auto()


# 0.879... = scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
STD_DEV_CONST = 0.87962566103423978


def _compute_fans(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) == 2:
        fan_in, fan_out = shape[0], shape[1]
    elif len(shape) > 2:
        receptive_field = np.prod(shape[2:])
        fan_in = shape[1] * receptive_field
        fan_out = shape[0] * receptive_field
    else:
        raise ValueError(f"The length of 'shape' must be greater or equal to 2, it is {len(shape)}.")
    return fan_in, fan_out


def _generate_distribution(shape: tuple[int, ...], scale: float, mode: DistributionModeEnum,
                           distribution: ProbabilisticDistributionEnum, dtype: np.dtype) -> np.ndarray:
    fan_in, fan_out = _compute_fans(shape)

    match mode:
        case DistributionModeEnum.FAN_IN:
            scale /= max(1., fan_in)
        case DistributionModeEnum.FAN_OUT:
            scale /= max(1., fan_out)
        case DistributionModeEnum.FAN_AVG:
            scale /= max(1., float(fan_in + fan_out) / 2)
        case _:
            raise NotImplementedError(f"mode: \'{mode}\' not implemented")

    match distribution:
        case ProbabilisticDistributionEnum.NORMAL:
            stddev: float = np.sqrt(scale) / STD_DEV_CONST
            # Truncated normal distribution [-2*stddev, 2*stddev]
            x = stats.truncnorm(-2 * stddev, 2 * stddev, loc=0, scale=stddev).rvs(shape).astype(dtype, copy=False)
        case ProbabilisticDistributionEnum.UNIFORM:
            limit = np.sqrt(3. * scale)
            x = np.random.uniform(-limit, limit, shape).astype(dtype, copy=False)
        case _:
            raise NotImplementedError(f"distribution: \'{distribution}\' not implemented")
    return x


def glorot_uniform(shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    return _generate_distribution(shape, 1.0, DistributionModeEnum.FAN_AVG,
                                  ProbabilisticDistributionEnum.UNIFORM, dtype)


def glorot_normal(shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    return _generate_distribution(shape, 1.0, DistributionModeEnum.FAN_AVG, ProbabilisticDistributionEnum.NORMAL, dtype)


def he_uniform(shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    return _generate_distribution(shape, 2.0, DistributionModeEnum.FAN_IN, ProbabilisticDistributionEnum.UNIFORM, dtype)


def he_normal(shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    return _generate_distribution(shape, 2.0, DistributionModeEnum.FAN_IN, ProbabilisticDistributionEnum.NORMAL, dtype)


def lecun_uniform(shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    return _generate_distribution(shape, 1.0, DistributionModeEnum.FAN_IN, ProbabilisticDistributionEnum.UNIFORM, dtype)


def lecun_normal(shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    return _generate_distribution(shape, 1.0, DistributionModeEnum.FAN_IN, ProbabilisticDistributionEnum.NORMAL, dtype)


def ones(shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    return np.ones(shape, dtype=dtype)


def zeros(shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    return np.zeros(shape, dtype=dtype)


type InitializerFunc = Callable[[tuple[int, ...], np.dtype], np.ndarray]
