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


class MemoryCache(dict):
    """
    Dictionary derived class that can use the provided factory function to
    obtain a default value for a missing key. It differs from defaultdict in:

    * The provided factory function receives key as a parameter (which allows
      the generated value to depend on the given key).

    * If disable() is called, the instances of this class will clear their
      already stored values and will not store the next ones.

    """
    _preserve_values = True

    def __init__(self, default_factory=None, **kwargs):
        super().__init__(self, **kwargs)
        self.default_factory = default_factory

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self.default_factory(key)
            if self._preserve_values:
                self[key] = ret
            return ret

    @classmethod
    def disable(cls):
        cls._preserve_values = False
        import gc
        for obj in gc.get_objects():
            if isinstance(obj, cls):
                obj.clear()

    @classmethod
    def enable(cls):
        cls._preserve_values = True
