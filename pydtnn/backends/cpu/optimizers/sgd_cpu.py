import numpy as np

from pydtnn.backends.cpu.optimizers import OptimizerCPU
from pydtnn.optimizers import SGD

from pydtnn.backends.cpu.layers import LayerCPU


class SGDCPU(OptimizerCPU, SGD):

    def initialize(self, list_layers: list[LayerCPU]) -> None:

        for layer in list_layers:
            list_grad_vars = list(layer.grad_vars.keys())
            if len(list_grad_vars) != 0:
                self.context[layer] = dict[str, np.ndarray]()
                for w_ in list_grad_vars:
                    w: np.ndarray = getattr(layer, w_)
                    self.context[layer]["velocity_%s" % w_] = np.zeros_like(w, dtype=layer.model.dtype)

    def update(self, layer: LayerCPU) -> None:
        for w_, dw_ in layer.grad_vars.items():
            w, dw = getattr(layer, w_), getattr(layer, dw_)
            velocity: np.ndarray = self.context[layer]["velocity_%s" % w_]
            w: np.ndarray
            dw: np.ndarray

            if not (self.are_all_zeros(velocity) and self.are_all_zeros(w) and self.are_all_zeros(dw)):
                # NOTE: The operations are unrolled in order to reduce the memory consumed by intermediate copies of the variables during the operations.

                # velocity = self.momentum * velocity + dw
                # NOTE/ Future FIXME: This will raise an error if the model is working in "int8" due is trying to assing a float64 value into a int8 ndarray.
                velocity *= self.momentum
                velocity += dw

                # if self.nesterov:
                #    w -= self.learning_rate * (self.decay * w + dw + self.momentum * velocity)
                # else:
                #    w -= self.learning_rate * (self.decay * w + velocity)
                if self.nesterov:
                    v = velocity * self.momentum
                    v += dw
                else:
                    v = velocity
                _w = w * self.decay
                _w += v
                _w *= self.learning_rate
                w -= _w
            # else: continue
