"""Provides representation and visualization utilities for PyDTNN models."""

import logging

from pydtnn import utils
from pydtnn.model.layers import Layers
from pydtnn.utils import header, term
from pydtnn.utils.constants import Array

__all__ = ("Repr",)

logger = logging.getLogger(__name__)


class Repr[T: Array](Layers[T]):  # noqa: D101 (generics not detected)
    """Mixin class providing string representation and summary visualization for models."""

    def _show_props(self) -> dict:  # noqa: C901
        """Collects model properties for representation and summary.

        Returns:
            dict: A dictionary containing model metadata and resource usage.
        """
        props = {}

        if self.model_name:
            props["name"] = self.model_name

        if self.dataset_name:
            props["dataset"] = self.dataset_name

        if self.nparams > 0:
            props["params"] = self.nparams

        if self.memory_used > 0:
            memory = utils.convert_size_bytes(self.memory_used) if self.memory_used > 0 else ""
            tmp_memory = (
                f"{utils.convert_size_bytes(self.tmp_memory_used)} tmp"
                if self.tmp_memory_used > 0
                else ""
            )
            if memory and tmp_memory:
                memory = f"{memory} ({tmp_memory})"
            elif tmp_memory:
                memory = tmp_memory
            if memory:
                props["memory"] = memory

        if self.loss_func:
            loss_memory = (
                utils.convert_size_bytes(self.loss_func.memory_used)
                if self.loss_func.memory_used > 0
                else ""
            )
            loss_tmp_memory = (
                f"{utils.convert_size_bytes(self.loss_func.tmp_memory_used)} tmp"
                if self.loss_func.tmp_memory_used > 0
                else ""
            )
            if loss_memory and loss_tmp_memory:
                loss_memory = f"{loss_memory} ({loss_tmp_memory})"
            elif loss_tmp_memory:
                loss_memory = loss_tmp_memory
            if loss_memory:
                props["loss-memory"] = loss_memory

        if self.metrics_funcs:
            metrics_size = 0
            metrics_temp_size = 0
            for metric in self.metrics_funcs:
                metrics_size += metric.memory_used
                metrics_temp_size += metric.tmp_memory_used
            metrics_memory = utils.convert_size_bytes(metrics_size) if metrics_size > 0 else ""
            metrics_tmp_memory = (
                f"{utils.convert_size_bytes(metrics_temp_size)} tmp"
                if metrics_temp_size > 0
                else ""
            )
            if metrics_memory and metrics_tmp_memory:
                metrics_memory = f"{metrics_memory} ({metrics_tmp_memory})"
            elif metrics_tmp_memory:
                metrics_memory = metrics_tmp_memory
            if metrics_memory:
                props["metrics-memory"] = metrics_memory

        if self.optimizer:
            optimizer_memory = (
                utils.convert_size_bytes(self.optimizer.memory_used)
                if self.optimizer.memory_used > 0
                else ""
            )
            optimizer_tmp_memory = (
                f"{utils.convert_size_bytes(self.optimizer.tmp_memory_used)} tmp"
                if self.optimizer.tmp_memory_used > 0
                else ""
            )
            if optimizer_memory and optimizer_tmp_memory:
                optimizer_memory = f"{optimizer_memory} ({optimizer_tmp_memory})"
            elif optimizer_tmp_memory:
                optimizer_memory = optimizer_tmp_memory
            if optimizer_memory:
                props["optimizer-memory"] = optimizer_memory

        if self.schedulers:
            schedulers_size = 0
            schedulers_temp_size = 0
            for scheduler in self.schedulers:
                schedulers_size += scheduler.memory_used
                schedulers_temp_size += scheduler.tmp_memory_used
            schedulers_memory = (
                utils.convert_size_bytes(schedulers_size) if schedulers_size > 0 else ""
            )
            schedulers_tmp_memory = (
                f"{utils.convert_size_bytes(schedulers_temp_size)} tmp"
                if schedulers_temp_size > 0
                else ""
            )
            if schedulers_memory and schedulers_tmp_memory:
                schedulers_memory = f"{schedulers_memory} ({schedulers_tmp_memory})"
            elif schedulers_tmp_memory:
                schedulers_memory = schedulers_tmp_memory
            if schedulers_memory:
                props["schedulers-memory"] = schedulers_memory

        if self.layers:
            props["input"] = self.layers[0].shape
            props["output"] = self.layers[-1].shape
            props["batch-size"] = self.batch_size
            props["layers"] = len(self.get_all_layers())

        return props

    def __repr__(self) -> str:
        """Returns a concise string representation of the model instance."""
        name = self.__class__.__name__
        props = " ".join(f"{key}={value!r}" for key, value in self._show_props().items())
        return f"<{name} {props}>" if props else f"<{name}>"

    def show_layers(self) -> None:
        """Logs a formatted table of all layers and their properties to the logger."""
        struct: dict[str, int] = {}
        all_props = {layer.id: layer._show_props() for layer in self.get_all_layers()}

        # Calculate headers and sizes
        for props in sorted(all_props.values(), key=lambda props: (-len(props), *props)):
            for key, value in props.items():
                struct[key] = max(struct.get(key, len(key)), len(str(value)))

        # Add header padding
        for key, size in struct.items():
            struct[key] += 2

        # Generate separator
        sep = []
        for key, size in struct.items():
            sep.append(term.BOX_H * size)
        tsep = f"{term.BOX_TL}{term.BOX_T.join(sep)}{term.BOX_TR}"
        csep = f"{term.BOX_L}{term.BOX_C.join(sep)}{term.BOX_R}"
        bsep = f"{term.BOX_BL}{term.BOX_B.join(sep)}{term.BOX_BR}"

        # Show header
        _show = [""]
        _show.append(tsep)
        _show.append("")
        for key, size in struct.items():
            _show[-1] += (
                f"{term.BOX_V}{term.BOLD}{key.replace('-', ' ').capitalize():^{size}s}{term.RESET}"
            )
        _show[-1] += term.BOX_V

        # Show layers
        top_layers = {layer.id for layer in self.layers}
        for layer_id, props in all_props.items():
            if layer_id in top_layers:
                _show.append(csep)
            _show.append("")
            for key, size in struct.items():
                value = props.get(key, "")
                _show[-1] += f"{term.BOX_V}{str(value):^{size}s}"
            _show[-1] += term.BOX_V
        _show.append(bsep)
        logger.info("\n".join(_show))

    def show_model(self) -> None:
        """Logs a summary of the model configuration to the logger."""
        key = "Model Summary"
        props = self._show_props()
        size = max(map(len, props))

        header(key)
        _show = []
        for key, value in props.items():
            _show.append(f"  {key:{size}s}: {value}")
        logger.info("\n".join(_show))

    def show(self) -> None:
        """Displays both the model summary and the detailed layer table."""
        self.show_model()
        self.show_layers()

    def print_in_convdirect_format(self) -> None:
        """Logs layer information formatted for ConvDirect compatibility."""
        line = "#l\tkn\two\tho\tt\tkh\tkw\tci\twi\thi"
        logger.info(line)
        for layer in self.layers:
            layer.print_in_convdirect_format()
