import numpy as np

from pydtnn.backends.cpu.optimizers.optimizer import OptimizerCPU
from pydtnn.optimizers.nadam import Nadam
from pydtnn.backends.cpu.layers.layer import LayerCPU


class NadamCPU(Nadam[np.ndarray], OptimizerCPU):

    def initialize(self, list_layers: list[LayerCPU]) -> None:

        for layer in list_layers:
            self.context[layer.id] = dict[str, int | np.ndarray]()
            self.context[layer.id]["it"] = 0

            for w_ in layer.grad_vars.keys():
                w: np.ndarray = getattr(layer, w_)
                self.context[layer.id]["m_%s" % w_] = np.zeros_like(w, dtype=layer.model.dtype, order="C")
                self.context[layer.id]["v_%s" % w_] = np.zeros_like(w, dtype=layer.model.dtype, order="C")

    def update(self, layer: LayerCPU) -> None:
        self.context[layer.id]["it"] += 1
        it: int = self.context[layer.id]["it"]  # type: ignore 

        for w_, dw_ in layer.grad_vars.items():
            w, dw = getattr(layer, w_), getattr(layer, dw_)
            w: np.ndarray
            dw: np.ndarray
            # Momentum of the weight or bias of the given layer
            m: np.ndarray = self.context[layer.id]["m_%s" % w_]  # type: ignore 
            # Velocity of the weight or bias of the given layer
            v: np.ndarray = self.context[layer.id]["v_%s" % w_]  # type: ignore 

            if not (self.are_all_zeros(w) and self.are_all_zeros(dw) or self.are_all_zeros(m) or self.are_all_zeros(v)):

                # NOTE: The operations are unrolled in order to reduce the memory consumed by intermediate copies of the variables during the operations.
                inv_beta1 = (1 - self.beta1)
                inv_beta2 = (1 - self.beta2)

                # m = self.beta1 * m + (1 - self.beta1) * dw
                temp_dw: np.ndarray = np.multiply(inv_beta1, dw, dtype=self.dtype, order="C")

                np.multiply(m, self.beta1, out=m, 
                            dtype=self.dtype)
                np.add(m, temp_dw, out=m,
                       dtype=self.dtype)

                # v = self.beta2 * v + (1 - self.beta2) * dw ** 2
                temp_dw = np.pow(dw, 2, dtype=self.dtype, order="C")

                np.multiply(v, self.beta2, out=v,
                            dtype=self.dtype)
                np.multiply(temp_dw, inv_beta2, out=temp_dw, 
                            dtype=self.dtype)
                np.add(v, temp_dw, out=v, 
                        dtype=self.dtype)
                del temp_dw

                # mt = (m + (1 - self.beta1) * dw) / (1 - self.beta1 ** it)
                mt = np.multiply(inv_beta1, dw, dtype=self.dtype, order="C")
                
                np.divide(mt, (inv_beta1 ** it), out=mt, 
                          dtype=self.dtype)
                np.add(mt, m, out=mt, 
                       dtype=self.dtype)

                # vt = v / (1 - self.beta2 ** it)
                vt = np.divide(v, (inv_beta2 ** it), dtype=self.dtype, order="C")

                # w -= self.learning_rate * (self.decay * w + (mt / np.sqrt(vt + epsilon)))
                temp_w = np.multiply((self.learning_rate * self.decay), w, dtype=self.dtype, order="C")
                
                np.subtract(w, temp_w, out=w, 
                            dtype=self.dtype)
                del temp_w

                np.add(vt, self.epsilon, out=vt,
                       dtype=self.dtype)
                np.sqrt(vt, out=vt,
                       dtype=self.dtype)
                np.divide(mt, vt, out=mt,
                          dtype=self.dtype)
                np.multiply(mt, self.learning_rate, out=mt, 
                            dtype=self.dtype, order="C")
                np.subtract(w, mt, out=w,
                            dtype=self.dtype)
                del mt, vt
            # else: continue
