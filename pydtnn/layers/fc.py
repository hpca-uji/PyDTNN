from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pydtnn.activations.activation import Activation
from pydtnn.layers.layer import Layer
from pydtnn.utils.types import Array
from pydtnn.initializers import InitializerFunc, glorot_uniform, zeros
from pydtnn.utils.types import ArrayShape


class FC[T: Array](Layer[T]):
    weights: T

    def __init__(self, shape: ArrayShape = (1,),
                 activation: "type[Activation] | None" = None,
                 use_bias=True,
                 weights_initializer: InitializerFunc = glorot_uniform,
                 biases_initializer: InitializerFunc = zeros):
        super().__init__(shape)
        self.act = activation
        self.use_bias = use_bias
        self.weights_initializer = weights_initializer
        self.biases_initializer = biases_initializer
        self.grad_vars = {"weights": "dw"}
        if self.use_bias:
            self.grad_vars["biases"] = "db"

    def show(self, attrs="") -> None:
        super().show("|{:^19s}|{:^37s}|".format(str(self.weights.shape), ""))
