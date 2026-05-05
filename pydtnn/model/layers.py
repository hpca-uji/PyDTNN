import logging
import operator
from collections import abc
from functools import reduce
from warnings import warn

from pydtnn.abstract.layerable import Layerable
from pydtnn.activations.relu import Relu
from pydtnn.backends.fuse.layers.layer import LayerFuse as FusedLayerMixIn
from pydtnn.backends.fuse.layers.layer import select as select_fuse_layer
from pydtnn.layers.batch_normalization import BatchNormalization
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.model.utils import Utils
from pydtnn.utils.constants import Array

__all__ = (
    "Layers",
)

logger = logging.getLogger(__name__)


class Layers[T: Array](Utils[T]):

    def add(self, layer: Layerable[T]) -> None:
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
        for layer in list_layers:
            self.add(layer)

    def get_all_layers(self, from_layers: list[Layerable[T]] | None = None) -> list[Layerable[T]]:
        if from_layers is None:
            from_layers = self.layers
        this_recursion_layers = []
        for layer in from_layers:
            this_recursion_layers.append(layer)
            children = layer.children
            this_recursion_layers += self.get_all_layers(children)
        return this_recursion_layers

    def _select_fusion_3(self, fused_layers: list) -> tuple[str | None, list[Layerable | FusedLayerMixIn | None]]:
        layer2 = fused_layers[-1] if len(fused_layers) > 0 else None
        layer1 = fused_layers[-2] if len(fused_layers) > 1 else None
        layer0 = fused_layers[-3] if len(fused_layers) > 2 else None

        layer_name = None

        match (layer0, layer1, layer2):
            case (_, FusedLayerMixIn(), _): pass  # else: layer_name = None
            case (Conv2D(), BatchNormalization(), Relu()):
                if self.enable_fused_conv_bn_relu:
                    layer_name = "conv_2d_batch_normalization_relu"
                # else: layer_name = None
            case _: pass  # else: layer_name = None

        return layer_name, [layer0, layer1, layer2]

    def _select_fusion_2(self, fused_layers: list) -> tuple[str | None, list[Layerable | FusedLayerMixIn | None]]:
        layer2 = fused_layers[-1] if len(fused_layers) > 0 else None
        layer1 = fused_layers[-2] if len(fused_layers) > 1 else None

        layer_name = None

        match (layer1, layer2):
            case (FusedLayerMixIn(), _): pass
            case (Conv2D(), BatchNormalization()):
                if self.enable_fused_conv_bn:
                    layer_name = "conv_2d_batch_normalization"
            case (Conv2D(), Relu()):
                if self.enable_fused_conv_relu:
                    layer_name = "conv_2d_relu"
            case (BatchNormalization(), Relu()):
                if self.enable_fused_bn_relu:
                    layer_name = "batch_normalization_relu"
            case _: pass

        return layer_name, [layer1, layer2]

    def _layer_fusion(self, layers: list[Layerable], switch_fusion: abc.Callable) -> None:
        i = 0
        while i < len(layers):
            curr_layer = layers[i]

            # Recurse if layer group
            for j, p in enumerate(curr_layer.paths):
                self._layer_fusion(curr_layer.paths[j], switch_fusion)

            layer_name, layers_to_fuse = switch_fusion(layers[:i])

            if layer_name:
                dict_params = reduce(operator.or_, (layer.__dict__ for layer in reversed(layers_to_fuse)))
                logger.info(f"Fusing {' + '.join(map(lambda layer: layer.name_with_id, layers_to_fuse))}")
                fused_layer = select_fuse_layer(layer_name)

                new_curr_layer = fused_layer(from_parent=dict_params)  # type: ignore (it's okay)
                new_curr_layer._init_backend_with_model(self)
                new_curr_layer.__dict__.update(dict_params)
                try:
                    new_curr_layer._model_init(prev_shape=layers_to_fuse[0].prev_shape, x=layers_to_fuse[0].x)
                except Exception as e:
                    warn_text = f"Aborted fusion, {e}"
                    logger.warning(warn_text)
                    warn(warn_text, RuntimeWarning)
                else:
                    start = i - len(layers_to_fuse)
                    del layers[start: i]
                    layers.insert(start, new_curr_layer)
                    i -= len(layers_to_fuse)
            i += 1

    def _apply_layer_fusion(self):
        """ Apply layer fusion in a recursive manner """

        if not self.enable_cudnn and any([self.enable_fused_bn_relu, self.enable_fused_conv_relu, self.enable_fused_conv_bn, self.enable_fused_conv_bn_relu]):
            # NOTE: 1st the 3 layers fusion, then the rest:
            self.backend = f"layers:fuse;{self.backend}"
            self._layer_fusion(self.layers, self._select_fusion_3)
            self._layer_fusion(self.layers, self._select_fusion_2)
