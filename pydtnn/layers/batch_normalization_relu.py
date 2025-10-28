from pydtnn.layers.batch_normalization import BatchNormalization
from pydtnn.utils.types import Array

class BatchNormalizationRelu[T: Array](BatchNormalization[T]):

    def __init__(self, *args, **kwargs):
        from_parent = kwargs.pop("from_parent", None)
        if from_parent is None:
            super().__init__(*args, **kwargs)
        else:
            from_parent.__dict__.pop("forward", None)
            self.__dict__.update(from_parent.__dict__)
