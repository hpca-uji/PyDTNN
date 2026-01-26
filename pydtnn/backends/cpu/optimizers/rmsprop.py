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
                    cache = np.zeros(w.shape, dtype=layer.model.dtype, order="C")
                    self.actual_size += cache.size
                    
                    if not self.model.use_memory_pool:
                        temp: np.ndarray = np.zeros(w.shape, dtype=layer.model.dtype, order="C")
                    else:
                        temp: np.ndarray = None  # type: ignore (it will be initialized later)
                    self.temp_size += int(np.prod(w.shape))

                    self.context[layer.id]["cache_%s" % w_] = cache
                    self.context[layer.id]["temp_%s" % w_] = temp
                    
                    self.actual_size += self.temp_size
            # else: continue
    # ----

    def post_initialize(self) -> None:
        super().post_initialize()

        for layer_id in self.context.keys():
            for key in self.context[layer_id].keys():
                if "temp_" in key:
                    w_ = key.split("temp_")[-1]
                    w_shape = self.context[layer_id]["cache_%s" % w_].shape # type: ignore (it is correct)
                    w_shape = self.context[layer_id][key] = self.model.memory_pool.get_ndarray(w_shape)
        # - end for
        self.model.memory_pool.free_memory(self.temp_size)
    # ---

    def update(self, layer: LayerCPU) -> None:
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
                np.power(dw, 2, dtype=self.dtype, order="C", out = temp)
                np.multiply(temp, (1 - self.rho), out=temp,
                        dtype=self.dtype)
                np.add(cache, temp, out=cache,
                       dtype=self.dtype)
                
                # w -= self.learning_rate * (self.decay * w + (dw / np.sqrt(cache + self.epsilon))) ==>
                # w -= (self.learning_rate * self.decay) * w + self.learning_rate * (dw / np.sqrt(cache + self.epsilon)))

                # w -= (self.learning_rate * self.decay) * w 
                np.multiply((self.learning_rate * self.decay), w, dtype=self.dtype, order="C", out=temp)
                np.subtract(w, temp, out=w, 
                            dtype=self.dtype)
                
                # w -= self.learning_rate * (dw / np.sqrt(cache + self.epsilon)))
                np.add(cache, self.epsilon, dtype=self.dtype, order="C", out=temp)
                np.sqrt(temp, out=temp,
                        dtype=self.dtype)
                np.divide(dw, temp, dtype=self.dtype, order="C", out=temp)
                np.multiply(temp, self.learning_rate, out=temp,
                            dtype=self.dtype)
                np.subtract(w, temp, out=w,
                            dtype=self.dtype)
            # else: continue
    # ----
