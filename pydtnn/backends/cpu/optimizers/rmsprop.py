import numpy as np

from pydtnn.backends.cpu.optimizers.optimizer import OptimizerCPU
from pydtnn.optimizers.rmsprop import RMSProp

from pydtnn.backends.cpu.layers.layer import LayerCPU


class RMSPropCPU(RMSProp[np.ndarray], OptimizerCPU):

    def initialize(self, list_layers: list[LayerCPU]) -> None:

        for layer in list_layers:
            list_grad_vars = list(layer.grad_vars.keys())

            if len(list_grad_vars) != 0:
                self.context[layer.id] = dict[str, np.ndarray]()  # type: ignore 
                for w_ in list_grad_vars:
                    w: np.ndarray = getattr(layer, w_)
                    self.context[layer.id]["cache_%s" % w_] = np.zeros_like(w, dtype=layer.model.dtype, order="C")

    def update(self, layer: LayerCPU) -> None:
        for w_, dw_ in layer.grad_vars.items():
            w, dw = getattr(layer, w_), getattr(layer, dw_)
            cache: np.ndarray = self.context[layer.id]["cache_%s" % w_]  # type: ignore 
            w: np.ndarray
            dw: np.ndarray

            if not (self.are_all_zeros(w) and self.are_all_zeros(dw) and self.are_all_zeros(cache)):
                # NOTE: The operations are unrolled in order to reduce the memory consumed by intermediate copies of the variables during the operations.

                # cache = self.rho * cache + (1 - self.rho) * dw ** 2
                np.multiply(cache, self.rho, out=cache, 
                            dtype=self.dtype)
                temp_dw = np.power(dw, 2, dtype=self.dtype, order="C")
                np.multiply(temp_dw, (1 - self.rho), out=temp_dw, 
                        dtype=self.dtype)
                np.add(cache, temp_dw, out=cache,
                       dtype=self.dtype)
                
                # w -= self.learning_rate * (self.decay * w + (dw / np.sqrt(cache + self.epsilon)))
                temp_w = np.multiply((self.learning_rate * self.decay), w, dtype=self.dtype, order="C")
                np.subtract(w, temp_w, out=w, 
                            dtype=self.dtype)
                del temp_w

                temp_cache = np.add(cache, self.epsilon, dtype=self.dtype, order="C")
                np.sqrt(temp_cache, out=temp_cache,
                        dtype=self.dtype)
                temp_dw = np.divide(dw, temp_cache, dtype=self.dtype, order="C")
                del temp_cache

                np.multiply(temp_dw, self.learning_rate, out=temp_dw,
                            dtype=self.dtype)
                np.subtract(w, temp_dw, out=w,
                            dtype=self.dtype)
                del temp_dw
            # else: continue
