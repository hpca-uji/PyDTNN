"""
Common methods and properties for various unitary tests
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
#  with this program. If not, see <https://www.gnu.org/licenses/>.
#

import sys


# @warning: must be a function, don't use a @property decorator
def verbose_test():
    """Returns True if unittest has been called with -v or --verbose options."""
    return '-v' in sys.argv or '--verbose' in sys.argv


class D:
    def __init__(self, b=1, c=1, h=128, w=100, kn=1, kh=16, kw=10, vpadding=1, hpadding=1,
                 vstride=1, hstride=1, vdilation=1, hdilation=1):
        self.b = b  # Batch size
        self.c = c  # Channels per layer
        self.h = h  # Layers height
        self.w = w  # Layers width
        self.kn = kn  # Number of filters
        self.kh = kh  # Filters weights height
        self.kw = kw  # Filters weights width
        self.vpadding = vpadding  # Vertical padding
        self.hpadding = hpadding  # Horizontal padding
        self.vstride = vstride  # Vertical stride
        self.hstride = hstride  # Horizontal stride
        self.vdilation = vdilation  # Vertical dilation
        self.hdilation = hdilation  # Horizontal dilation

    @property
    def ho(self):
        return (self.h + 2 * self.vpadding - self.vdilation * (self.kh - 1) - 1) // self.vstride + 1

    @property
    def wo(self):
        return (self.w + 2 * self.hpadding - self.hdilation * (self.kw - 1) - 1) // self.hstride + 1

    def __repr__(self):
        return f"""\
x, weights, and y parameters:
  (b, c, h, w)    = {self.b} {self.c} {self.h} {self.w}
  (kn, c, kh, kw) = {self.kn} {self.c} {self.kh} {self.kw}
  (kn, b, ho, wo) = {self.kn} {self.b} {self.ho} {self.wo}
  padding         = {self.vpadding} {self.hpadding}
  stride          = {self.vstride} {self.hstride}
  dilation        = {self.vdilation} {self.hdilation}
"""


alexnet_layers = [
    # AlexNet Cifar
    D(64, 3, 32, 32, 64, 3, 3, 1, 1, 2, 2, 1, 1),
    D(64, 64, 8, 8, 192, 3, 3, 1, 1, 1, 1, 1, 1),
    D(64, 192, 4, 4, 384, 3, 3, 1, 1, 1, 1, 1, 1),
    D(64, 384, 4, 4, 256, 3, 3, 1, 1, 1, 1, 1, 1),
    D(64, 256, 4, 4, 256, 3, 3, 1, 1, 1, 1, 1, 1),
    # AlexNet ImageNet
    D(64, 3, 227, 227, 96, 11, 11, 1, 1, 4, 4, 1, 1),
    D(64, 96, 27, 27, 256, 5, 5, 1, 1, 1, 1, 1, 1),
    D(64, 256, 13, 13, 384, 3, 3, 1, 1, 1, 1, 1, 1),
    D(64, 384, 13, 13, 384, 3, 3, 1, 1, 1, 1, 1, 1),
    D(64, 384, 13, 13, 256, 3, 3, 1, 1, 1, 1, 1, 1),
]
