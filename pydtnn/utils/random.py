

import threading
import weakref

import numpy as np


class SafeGenerator:
    def __init__(self, seed=0) -> None:
        self._generators = weakref.WeakKeyDictionary[threading.Thread, np.random.Generator]()
        self.seed(seed)

    def seed(self, seed) -> None:
        self._seed = seed
        self._generators.clear()

    @property
    def _generator(self) -> np.random.Generator:
        thread = threading.current_thread()

        if thread not in self._generators:
            self._generators[thread] = np.random.default_rng(self._seed)

        return self._generators[thread]

    def __getattr__(self, key: str):
        return getattr(self._generator, key)

    def shuffle(self, x: list | np.ndarray, axis=0) ->  None:
        """Modify an array or sequence in-place by shuffling its contents."""
        # NOTE: CuPy does not provide an implementation

        if isinstance(x, list):
            if axis != 0:
                raise ValueError("lists only support shuffle on axis 0!")
            idx = self.permutation(len(x))
            tmp = [x[i] for i in idx]
            x[:] = tmp
            return

        idx = self.permutation(x.shape[axis])
        slc = [slice(None)] * x.ndim
        slc[axis] = idx
        x[:] = x[tuple(slc)]


_global = SafeGenerator()


def __getattr__(key):
    return getattr(_global, key)
