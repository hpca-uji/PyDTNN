import logging

from pydtnn import utils
from pydtnn.model_parts.layers import Layers
from pydtnn.utils.constants import Array

logger = logging.getLogger(__name__)


class Print[T: Array](Layers[T]):

    def _show_props(self) -> dict:
        props = {}

        if self.model_name:
            props["name"] = self.model_name

        if self.dataset_name:
            props["dataset"] = self.dataset_name

        if self.nparams > 0:
            props["params"] = self.nparams

        if self.memory_used > 0:
            memory = utils.convert_size_bytes(self.memory_used)
            if self.tmp_memory_used > 0:
                tmp_memory = utils.convert_size_bytes(self.tmp_memory_used)
                memory = f"{memory} ({tmp_memory} tmp)"
            props["memory"] = memory

        if self.optimizer:
            optimizer_memory = utils.convert_size_bytes(self.optimizer.memory_used)
            if self.optimizer.tmp_memory_used > 0:
                optimizer_tmp_memory = utils.convert_size_bytes(self.optimizer.tmp_memory_used)
                optimizer_memory = f"{optimizer_memory} ({optimizer_tmp_memory} tmp)"
            props["optimizer-memory"] = optimizer_memory

        if self.loss_func:
            loss_memory = utils.convert_size_bytes(self.loss_func.memory_used)
            if self.loss_func.tmp_memory_used > 0:
                loss_tmp_memory = utils.convert_size_bytes(self.loss_func.tmp_memory_used)
                loss_memory = f"{loss_memory} ({loss_tmp_memory} tmp)"
            props["loss-memory"] = loss_memory

        if self.metrics_funcs:
            metrics_size = 0
            metric_temp_size = 0
            for metric in self.metrics_funcs:
                metrics_size += metric.memory_used
                metric_temp_size += metric.tmp_memory_used
            metrics_memory = utils.convert_size_bytes(metrics_size)
            if metric_temp_size > 0:
                metrics_tmp_memory = utils.convert_size_bytes(metric_temp_size)
                metrics_memory = f"{metrics_memory} ({metrics_tmp_memory} tmp)"
            props["metrics-memory"] = metrics_memory

        if self.layers:
            props["input"] = self.layers[0].shape
            props["output"] = self.layers[-1].shape
            props["batch-size"] = self.batch_size
            props["layers"] = len(self.get_all_layers())

        return props

    def __repr__(self) -> str:
        props = " ".join(
            f"{key}={value!r}"
            for key, value in self._show_props().items()
        )

        return f"<{self.__class__.__name__} {props}>"

    def show_layers(self) -> None:
        struct: dict[str, int] = {}
        all_props = {
            layer.id: layer._show_props()
            for layer in self.get_all_layers()
        }

        # Calculate headers and sizes
        for props in sorted(all_props.values(), key=lambda props: (-len(props), *props)):
            for key, value in props.items():
                struct[key] = max(struct.get(key, len(key)), len(str(value)))

        # Add header padding
        for header, size in struct.items():
            struct[header] += 2

        # Generate separator
        sep = ""
        for header, size in struct.items():
            sep += "+" + "-" * size
        sep += "+"

        # Show header
        _show = [""]
        _show.append(sep)
        _show.append("")
        for header, size in struct.items():
            _show[-1] += (f"|{header.replace('-', ' ').capitalize():^{size}s}")
        _show[-1] += ("|")

        # Show layers
        top_layers = {layer.id for layer in self.layers}
        for layer_id, props in all_props.items():
            if layer_id in top_layers:
                _show.append(sep)
            _show.append("")
            for header, size in struct.items():
                value = props.get(header, "")
                _show[-1] += (f"|{str(value):^{size}s}")
            _show[-1] += ("|")
        _show.append(sep)
        logger.info('\n'.join(_show))

    def show_model(self) -> None:
        key: str = "Model Summary"
        _show = [""]
        _show.append(key)
        _show.append("=" * len(key))
        for key, value in self._show_props().items():
            _show.append(f"- {key.replace('-', ' ').capitalize()}: {value}")
        logger.info('\n'.join(_show))

    def show(self) -> None:
        self.show_model()
        self.show_layers()

    def print_in_convdirect_format(self) -> None:
        line = "#l\tkn\two\tho\tt\tkh\tkw\tci\twi\thi"
        logger.info(line)
        for layer in self.layers:
            layer.print_in_convdirect_format()
