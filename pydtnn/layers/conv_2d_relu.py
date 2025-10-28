from pydtnn.layers.conv_2d import Conv2D
from pydtnn.utils.types import Array

class Conv2DRelu[T: Array](Conv2D[T]):
    def __init__(self, *args, **kwargs):
        from_parent = kwargs.pop("from_parent", None)
        if from_parent is None:
            super().__init__(*args, **kwargs)
        else:
            # from_parent.__dict__.pop("forward", None)
            self.__dict__.update(from_parent.__dict__)
