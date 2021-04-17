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

from abc import ABC

from .abstract_block_layer import AbstractBlockLayer


class ConcatenationBlock(AbstractBlockLayer, ABC):

    def show(self, attrs=""):
        print(
            f"|{self.id:^7d}"
            f"|{(type(self).__name__.replace('Concatenation', 'Concat') + ' (%d-path)' % len(self.paths)):^26s}"
            f"|{'':9s}|{str(self.shape):^15s}|{'':19s}|{'':24s}|")
        for i, p in enumerate(self.paths):
            print(f"|{('Path %d' % i):^7s}|{'':^26s}|{'':9s}|{'':15s}|{'':19s}|{'':24s}|")
            for layer in p:
                layer.show()
