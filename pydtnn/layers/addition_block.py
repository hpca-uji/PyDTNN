from pydtnn.layers.abstract.block_layer import AbstractBlockLayer
from pydtnn.utils.constants import Array

class AdditionBlock[T: Array](AbstractBlockLayer[T]):

    def _show(self, attrs=""):
        print(f"|{self.id:^7d}|{(type(self).__name__ + ' (%d-path)' % len(self.paths)):^26s}|{'':9s}"
              f"|{str(self.shape):^15s}|{'':19s}|{'':37s}|")
        for i, p in enumerate(self.paths):
            print(f"|{('Path %d' % i):^7s}|{'':^26s}|{'':9s}|{'':15s}|{'':19s}|{'':37s}|")
            for layer in p:
                layer._show()
