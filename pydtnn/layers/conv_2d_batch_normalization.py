from pydtnn.layers.conv_2d import Conv2D
from pydtnn.layers.batch_normalization import BatchNormalization
from pydtnn.utils.types import Array

class Conv2DBatchNormalization[T: Array](Conv2D[T], BatchNormalization[T]):

    def __init__(self, *args, **kwargs):
        from_parent = kwargs.pop("from_parent", None)
        from_parent2 = kwargs.pop("from_parent2", None)
        if from_parent is None and from_parent2 is None:
            super().__init__(*args, **kwargs)
        else:
            # from_parent.__dict__.pop("forward", None)
            # from_parent2.__dict__.pop("forward", None)
            self.__dict__.update(from_parent.__dict__)
            self.__dict__.update(from_parent2.__dict__)
