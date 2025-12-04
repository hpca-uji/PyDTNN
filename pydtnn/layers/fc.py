from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pydtnn.activations.activation import Activation
from pydtnn.layers.layer import Layer
from pydtnn.utils.initializers import InitializerFunc, glorot_uniform, zeros
from pydtnn.utils.constants import Array, ArrayShape, Parameters


class FC[T: Array](Layer[T]):

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
        self.grad_vars = {Parameters.WEIGHTS: Parameters.DW}
        if self.use_bias:
            self.grad_vars[Parameters.BIASES] = Parameters.DB

    def initialize(self, prev_shape: ArrayShape, x: T | None) -> None:
        super().initialize(prev_shape, x)
        self.weights_shape = (*prev_shape, *self.shape)
