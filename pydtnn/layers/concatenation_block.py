from abc import ABC

from pydtnn.layers.abstract_block_layer import AbstractBlockLayer


class ConcatenationBlock(AbstractBlockLayer, ABC):

    def show(self, attrs="") -> None:
        print(
            f"|{self.id:^7d}"
            f"|{(type(self).__name__.replace('Concatenation', 'Concat') + ' (%d-path)' % len(self.paths)):^26s}"
            f"|{'':9s}|{str(self.shape):^15s}|{'':19s}|{'':37s}|")
        for i, p in enumerate(self.paths):
            print(f"|{('Path %d' % i):^7s}|{'':^26s}|{'':9s}|{'':15s}|{'':19s}|{'':37s}|")
            for layer in p:
                layer.show()
