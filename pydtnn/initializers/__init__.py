"""
PyDTNN initializers
"""

#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021 Universitat Jaume I
#
#  PyDTNN is free software: you can redistribute it and/or modify it under the
#  terms of the GNU General Public License as published by the Free Software
#  Foundation, either version 3 of the License, or (at your option) any later
#  version.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT
#  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
#  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
#  License for more details.
#
#  You should have received a copy of the GNU General Public License along
#  with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import numpy as np
import scipy.stats as stats


def _compute_fans(shape):
    if len(shape) == 2:
        fan_in, fan_out = shape[0], shape[1]
    elif len(shape) > 2:
        receptive_field = np.prod(shape[2:])
        fan_in = shape[1] * receptive_field
        fan_out = shape[0] * receptive_field
    else:
        raise ValueError(f"The length of 'shape' must be greater or equal to 2, it is {len(shape)}.")
    return fan_in, fan_out


def _generate_distribution(shape, scale, mode, distribution, dtype):
    fan_in, fan_out = _compute_fans(shape)
    if mode == 'fan_in':
        scale /= max(1., fan_in)
    elif mode == 'fan_out':
        scale /= max(1., fan_out)
    else:
        scale /= max(1., float(fan_in + fan_out) / 2)
    if distribution == 'normal':
        # 0.879... = scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        stddev = np.sqrt(scale) / .87962566103423978
        # Truncated normal distribution [-2*stddev, 2*stddev]
        x = stats.truncnorm(-2 * stddev, 2 * stddev, loc=0, scale=stddev).rvs(shape).astype(dtype)
    else:
        limit = np.sqrt(3. * scale)
        x = np.random.uniform(-limit, limit, shape).astype(dtype)
    return x


def glorot_uniform(shape, dtype):
    return _generate_distribution(shape, 1.0, "fan_avg", "uniform", dtype)


def glorot_normal(shape, dtype):
    return _generate_distribution(shape, 1.0, "fan_avg", "normal", dtype)


def he_uniform(shape, dtype):
    return _generate_distribution(shape, 2.0, "fan_in", "uniform", dtype)


def he_normal(shape, dtype):
    return _generate_distribution(shape, 2.0, "fan_in", "normal", dtype)


def lecun_uniform(shape, dtype):
    return _generate_distribution(shape, 1.0, "fan_in", "uniform", dtype)


def lecun_normal(shape, dtype):
    return _generate_distribution(shape, 1.0, "fan_in", "normal", dtype)


def ones(shape, dtype):
    return np.ones(shape).astype(dtype)


def zeros(shape, dtype):
    return np.zeros(shape).astype(dtype)
