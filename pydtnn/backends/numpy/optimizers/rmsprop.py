from pydtnn.libs import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np

from pydtnn.backends.numpy.optimizers.optimizer import OptimizerNumpy
from pydtnn.optimizers.rmsprop import RMSProp

from pydtnn.backends.numpy.layers.layer import LayerNumpy


class RMSPropNumpy(RMSProp[np.ndarray], OptimizerNumpy):

    def initialize(self, list_layers: list[LayerNumpy]) -> None:

        temp_memory_size = []

        for layer in list_layers:
            list_grad_vars = list(layer.grad_vars.keys())

            if len(list_grad_vars) != 0:
                self.context[layer.id] = dict[str, np.ndarray]()  # type: ignore
                for w_ in list_grad_vars:
                    w: np.ndarray = getattr(layer, w_)
                    cache = np.zeros(w.shape, dtype=layer.model.dtype)
                    temp = None
                    self.real_memory_size += cache.nbytes

                    temp_memory_size.append(int(np.prod(w.shape)) * self.model.dtype.itemsize)
                    self.context[layer.id]["cache_%s" % w_] = cache
                    self.context[layer.id]["temp_%s" % w_] = temp

        self.temp_memory_size += self.model.memory_cls._total(*temp_memory_size)
        self.real_memory_size += self.temp_memory_size
    # ----

    def post_initialize(self) -> None:
        super().post_initialize()
        for layer_id in self.context.keys():
            with self.model.memory:
                for key in self.context[layer_id].keys():
                    if "temp_" in key:
                        w_ = key.split("temp_")[-1]
                        w_shape = self.context[layer_id]["cache_%s" % w_].shape  # type: ignore (it is correct)
                        w_shape = self.context[layer_id][key] = self.model.memory.ndarray(w_shape, dtype=self.model.dtype)
        # - end for
    # ---

    def update(self, layer: LayerNumpy) -> None:
        for w_, dw_ in layer.grad_vars.items():
            w, dw = getattr(layer, w_), getattr(layer, dw_)
            cache: np.ndarray = self.context[layer.id]["cache_%s" % w_]  # type: ignore
            temp: np.ndarray = self.context[layer.id]["temp_%s" % w_]  # type: ignore
            w: np.ndarray
            dw: np.ndarray

            if not (self.are_all_zeros(w) and self.are_all_zeros(dw) and self.are_all_zeros(cache)):
                # NOTE: The operations are unrolled in order to reduce the memory consumed by intermediate copies of the variables during the operations.

                # cache = self.rho * cache + (1 - self.rho) * dw ** 2
                np.multiply(cache, self.rho, out=cache,
                            dtype=self.dtype)
                np.power(dw, 2, dtype=self.dtype, out=temp)
                np.multiply(temp, (1 - self.rho), out=temp,
                            dtype=self.dtype)
                np.add(cache, temp, out=cache,
                       dtype=self.dtype)

                # w -= self.learning_rate * (self.decay * w + (dw / np.sqrt(cache + self.epsilon))) ==>
                # w -= (self.learning_rate * self.decay) * w + self.learning_rate * (dw / np.sqrt(cache + self.epsilon)))

                # w -= (self.learning_rate * self.decay) * w
                np.multiply((self.learning_rate * self.decay), w, dtype=self.dtype, out=temp)
                np.subtract(w, temp, out=w,
                            dtype=self.dtype)

                # w -= self.learning_rate * (dw / np.sqrt(cache + self.epsilon)))
                np.add(cache, self.epsilon, dtype=self.dtype, out=temp)
                np.sqrt(temp, out=temp,
                        dtype=self.dtype)
                np.divide(dw, temp, dtype=self.dtype, out=temp)
                np.multiply(temp, self.learning_rate, out=temp,
                            dtype=self.dtype)
                np.subtract(w, temp, out=w,
                            dtype=self.dtype)
            # else: continue
    # ----
