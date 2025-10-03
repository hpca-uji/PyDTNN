import numpy as np

from pydtnn.backends.cpu.optimizers import OptimizerCPU
from pydtnn.optimizers import Adam
from pydtnn.backends.cpu.layers import LayerCPU


class AdamCPU(OptimizerCPU, Adam):

    def initialize(self, list_layers: list[LayerCPU]) -> None:

        for layer in list_layers:
            self.context[layer] = dict[str, int | np.ndarray]()
            self.context[layer]["it"] = 0

            for w_ in layer.grad_vars.keys():
                w: np.ndarray = getattr(layer, w_)
                self.context[layer]["m_%s" % w_] = np.zeros_like(w, dtype=layer.model.dtype)
                self.context[layer]["v_%s" % w_] = np.zeros_like(w, dtype=layer.model.dtype)

    def update(self, layer: LayerCPU) -> None:
        self.context[layer]["it"] += 1
        it: int = self.context[layer]["it"]

        for w_, dw_ in layer.grad_vars.items():
            w, dw = getattr(layer, w_), getattr(layer, dw_)
            w: np.ndarray
            dw: np.ndarray
            # Momentum of the weight or bias of the given layer
            m: np.ndarray = self.context[layer]["m_%s" % w_]
            # Velocity of the weight or bias of the given layer
            v: np.ndarray = self.context[layer]["v_%s" % w_]

            if not (self.are_all_zeros(w) and self.are_all_zeros(dw) and self.are_all_zeros(m) and self.are_all_zeros(v)):
                # NOTE: The operations are unrolled in order to reduce the memory consumed by intermediate copies of the variables during the operations.
                # m = self.beta1 * m + (1 - self.beta1) * dw
                m *= self.beta1
                m += (1 - self.beta1) * dw

                # v = self.beta2 * v + (1 - self.beta2) * dw ** 2
                v *= self.beta2
                _dw = dw ** 2
                _dw *= (1 - self.beta2)
                v += _dw

                mt: np.ndarray = m / (1 - self.beta1 ** it)
                vt: np.ndarray = v / (1 - self.beta2 ** it)

                # w -= self.learning_rate * (self.decay * w + (mt / np.sqrt(vt + self.epsilon)))
                _w = self.decay * w

                vt += self.epsilon
                np.sqrt(vt, out=vt)
                mt /= vt

                _w += mt
                _w *= self.learning_rate

                w -= _w
            # else: continue
