from abc import ABC

from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from pydtnn.activations import Activation
from .layer import Layer
from pydtnn.initializers import InitializerFunc, glorot_uniform, zeros


class FC(Layer, ABC):

    def __init__(self, shape: tuple[int,...] = (1,), 
                 activation: Optional["Activation"] = None, 
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
