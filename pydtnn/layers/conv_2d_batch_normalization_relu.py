from abc import ABC

from .conv_2d import Conv2D
from .batch_normalization import BatchNormalization


class Conv2DBatchNormalizationRelu(Conv2D, BatchNormalization, ABC):

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
