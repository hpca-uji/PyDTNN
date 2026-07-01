"""Module providing the Layers management class for the PyDTNN framework."""

import logging
import operator
from collections import abc
from functools import reduce

from pydtnn.abstract.layerable import Layerable
from pydtnn.activations.relu import Relu
from pydtnn.backends.fuse.layers import select as select_layer
from pydtnn.backends.fuse.layers.abstract.layer import LayerFuse
from pydtnn.layers.batch_normalization import BatchNormalization
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.model.utils import Utils
from pydtnn.utils.constants import Array

__all__ = ("Layers",)

logger = logging.getLogger(__name__)


class Layers[T: Array](Utils[T]):
    """Manages the collection and lifecycle of neural network layers."""

    def add(self, layer: Layerable[T]) -> None:
        """Adds a layer to the model and initializes its backend and parameters."""
        layer._init_backend_with_model(self)

        if self.layers:
            prev_shape = self.layers[-1].shape
            y = self.layers[-1].y
        else:
            prev_shape = ()
            y = None

        layer._model_init(prev_shape, y)

        self.nparams += layer.nparams
        # self.memory_used += layer.memory_used
        # self.tmp_memory_used += layer.tmp_memory_used
        self.layers.append(layer)

        if layer.act:
            self.add(layer.act())

    def add_layers(self, list_layers: abc.Sequence[Layerable[T]]) -> None:
        """Adds a sequence of layers to the model."""
        for layer in list_layers:
            self.add(layer)

    def get_all_layers(self, from_layers: list[Layerable[T]] | None = None) -> list[Layerable[T]]:
        """Recursively retrieves all layers, including nested children."""
        if from_layers is None:
            from_layers = self.layers
        this_recursion_layers = []
        for layer in from_layers:
            this_recursion_layers.append(layer)
            children = layer.children
            this_recursion_layers += self.get_all_layers(children)
        return this_recursion_layers

    def _select_fusion_3(
        self, fused_layers: list
    ) -> tuple[str | None, list[Layerable | LayerFuse | None]]:
        """Identifies potential 3-layer fusion patterns."""
        layer2 = fused_layers[-1] if len(fused_layers) > 0 else None
        layer1 = fused_layers[-2] if len(fused_layers) > 1 else None
        layer0 = fused_layers[-3] if len(fused_layers) > 2 else None

        layer_name = None

        match (layer0, layer1, layer2):
            case (_, LayerFuse(), _):
                pass  # skip fused
            case (Conv2D(), BatchNormalization(), Relu()):
                if self.enable_fused_conv_bn_relu:
                    layer_name = "conv_2d_batch_normalization_relu"

        return layer_name, [layer0, layer1, layer2]

    def _select_fusion_2(
        self, fused_layers: list
    ) -> tuple[str | None, list[Layerable | LayerFuse | None]]:
        """Identifies potential 2-layer fusion patterns."""
        layer2 = fused_layers[-1] if len(fused_layers) > 0 else None
        layer1 = fused_layers[-2] if len(fused_layers) > 1 else None

        layer_name = None

        match (layer1, layer2):
            case (LayerFuse(), _):
                pass  # skip fused
            case (Conv2D(), BatchNormalization()):
                if self.enable_fused_conv_bn:
                    layer_name = "conv_2d_batch_normalization"
            case (Conv2D(), Relu()):
                if self.enable_fused_conv_relu:
                    layer_name = "conv_2d_relu"
            case (BatchNormalization(), Relu()):
                if self.enable_fused_bn_relu:
                    layer_name = "batch_normalization_relu"

        return layer_name, [layer1, layer2]

    def _layer_fusion(self, layers: list[Layerable], switch_fusion: abc.Callable) -> None:
        """Performs layer fusion on the provided list using a specific fusion strategy."""
        i = 0
        while i < len(layers):
            curr_layer = layers[i]

            # Recurse if layer group
            for j, p in enumerate(curr_layer.paths):
                self._layer_fusion(curr_layer.paths[j], switch_fusion)

            # NOTE: i+1 include current layer, range(i+1) excludes end
            layer_name, layers_to_fuse = switch_fusion(layers[: i + 1])

            if layer_name:
                dict_params = reduce(
                    operator.or_, (layer.__dict__ for layer in reversed(layers_to_fuse))
                )
                memory_used = reduce(
                    operator.add, (layer.memory_used for layer in reversed(layers_to_fuse))
                )
                tmp_memory_used = reduce(
                    self.memory_cls._total,
                    (layer.tmp_memory_used for layer in reversed(layers_to_fuse)),
                )
                dict_params |= {"memory_used": memory_used, "tmp_memory_used": tmp_memory_used}
                logger.info(
                    f"Fusing {' + '.join(map(lambda layer: layer.name_with_id, layers_to_fuse))}"
                )
                fused_layer = select_layer(layer_name)

                new_curr_layer = fused_layer(from_parent=dict_params)  # type: ignore (it's okay)
                new_curr_layer._init_backend_with_model(self)
                new_curr_layer.__dict__.update(dict_params)
                try:
                    new_curr_layer._model_init(
                        prev_shape=layers_to_fuse[0].prev_shape, x=layers_to_fuse[0].x
                    )
                except Exception:
                    logger.warning("Aborted fusion", exc_info=True)
                else:
                    start = i + 1 - len(layers_to_fuse)
                    layers[start: i + 1] = [new_curr_layer]
                    i -= len(layers_to_fuse)
            i += 1

    def _apply_layer_fusion(self) -> None:
        """Apply layer fusion in a recursive manner"""

        if not self.enable_cudnn and any(
            [
                self.enable_fused_bn_relu,
                self.enable_fused_conv_relu,
                self.enable_fused_conv_bn,
                self.enable_fused_conv_bn_relu,
            ]
        ):
            # NOTE: 1st the 3 layers fusion, then the rest:
            self.backend = f"layers:fuse;{self.backend}"
            self._layer_fusion(self.layers, self._select_fusion_3)
            self._layer_fusion(self.layers, self._select_fusion_2)
