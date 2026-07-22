"""
PyDTNN parser

The parser in this module will be used by 'pydtnn_benchmark' to parse the
command line arguments.

And what is even more important, it will also be loaded by the Model class to
obtain default values to its non-mandatory attributes. This way, when a model
object is instantiated (even if it is not from 'pydtnn_benchmark') it will
initially have default values for all the attributes declared on the self.

If you want to define a new option, just declare it here. It will automatically
be available as a Model attribute.
"""

import argparse
import logging
import os
import re
import textwrap
from importlib import resources
from pathlib import PurePath
from typing import Sequence

import numpy as np

from pydtnn import cublas, drv, gpuarray, package_name
from pydtnn.abstract.base import Base
from pydtnn.datasets.abstract.base import Base as DatasetBase
from pydtnn.model.base import Base as ModelBase
from pydtnn.utils import read_dir
from pydtnn.utils.constants import NetworkAlgoEnum
from pydtnn.utils.gpu import get_gpus_per_node
from pydtnn.utils.tensor import TensorFormat

__all__ = (
    "ArgumentParser",
    "Namespace",
    "factor",
    "np_dtype",
)

logger = logging.getLogger(__name__)


def factor(_x: str) -> float:
    """Returns x, which must be 0.0 < x <= 1.0"""
    x = float(_x)
    if not (0.0 < x <= 1.0):
        raise ValueError("Provided value must be greater than 0.0 and less or equal to 1.0")
    return x


def np_dtype(x: str) -> np.dtype:
    """Returns a numpy object from a string representing the data type"""
    return np.dtype(getattr(np, x))


def csi(x: str) -> tuple[int, ...]:
    """Parse coma separated integers"""
    return tuple(map(int, filter(None, x.split(","))))


def csf(x: str) -> tuple[float, ...]:
    """Parse coma separated floats"""
    return tuple(map(float, filter(None, x.split(","))))


def css(x: str) -> tuple[str, ...]:
    """Parse coma separated strings"""
    return tuple(filter(None, x.split(",")))


def list_modules(path: str) -> list[str]:
    """List public modules in package's path"""
    return [
        PurePath(resource.name).stem
        for resource in resources.files(package_name).joinpath(path).iterdir()
        if resource.is_file()
        and not resource.name.startswith("_")
        and resource.name.endswith(".py")
    ]


def _get_mpi_processes() -> int:
    """Returns the number of MPI processes from the environment."""
    try:
        from pympi import MPI  # type: ignore
    except Exception:
        mpi_processes = 1
    else:
        mpi_processes = MPI.COMM_WORLD.Get_size()
    return mpi_processes


def _get_threads_per_process() -> int:
    """Returns the number of OpenMP threads per process."""
    #  From IBM OpenMP documentation: If you do not set OMP_NUM_THREADS, the number of processors available is the
    #  default value to form a new team for the first encountered parallel construct.
    threads_per_process = os.environ.get("OMP_NUM_THREADS", os.process_cpu_count())
    return int(threads_per_process)


def _get_mpi_protocol() -> str | None:
    """Returns the MPI communication protocol string."""
    try:
        from pydtnn.libs.mpi.rc import proto, ssl
    except Exception:
        proto = None
        ssl = None
    if proto:
        protocol = str(proto)
    else:
        protocol = "native"
    if proto and ssl:
        protocol = f"{protocol}+tls"
    return protocol


def _get_mpi_server() -> str | None:
    """Returns the MPI server address."""
    try:
        from pydtnn.libs.mpi.rc import addr
    except Exception:
        addr = None
    return addr


def _get_mpi_port() -> int | None:
    """Returns the MPI port number."""
    try:
        from pydtnn.libs.mpi.rc import port
    except Exception:
        port = None
    return port


def _get_use_cudnn() -> bool:
    """Get if cudnn is enabled."""
    return gpuarray is not None and drv is not None and cublas is not None


class Namespace(argparse.Namespace):
    """Custom namespace for storing parsed arguments and group information."""

    def __str__(self) -> str:
        """Returns a formatted string representation of the namespace arguments."""
        lines = []
        for group in self.groups:
            indent = ""
            length = 0
            if group.title not in ("positional arguments", "optional arguments"):
                indent = "  "
                lines.append("")
                lines.append(group.title)
                if group.description is not None:
                    lines.append(group.description)
            for action in group._group_actions:
                if action.default == "==SUPPRESS==":
                    continue
                option_string = f"{action.option_strings[0].replace('--', '')}"
                if len(option_string) > length:
                    length = len(option_string)
            for action in group._group_actions:
                if action.default == "==SUPPRESS==":
                    continue
                option_string = f"{action.option_strings[0].replace('--', '')}"
                tab = " " * (length - len(option_string))
                lines.append(f"{indent}{option_string}{tab}: {getattr(self, action.dest)}")
        lines.append("")
        return "\n".join(lines)


class ArgumentParser(argparse.ArgumentParser):
    """Custom argument parser for PyDTNN configuration."""

    def __init__(self) -> None:
        """Initializes the parser with all supported PyDTNN configuration arguments."""
        super().__init__(
            description="Trains or evaluates a neural network using PyDTNN.",
            epilog=(
                "Example scripts that call this program for training"
                " and evaluating different neural network models with"
                " different datasets are available at 'scripts'."
            ),
        )

        # Parser and the supported arguments with their default values
        # (argparse.SUPPRESS is used to avoid showing them on the message)

        # Model
        self._optionals.title = "Model options"
        models = list_modules("models")
        self.add_argument(
            "--model",
            dest="model_name",
            type=str,
            choices=models,
            default="simplecnn",
            help=(
                f"Neural network model: {', '.join(map(repr, models[:3]))}, etc."
                f" Default: {ModelBase.model_name!r}."
            ),
        )
        backends = read_dir("backends")
        self.add_argument(
            "--backend",
            type=str,
            default=ModelBase.backend,
            help=(
                "Backend selection priority."
                " Format: [module[,module[,...]]:]backend[,backend[,...]][;...]."
                " Example: 'all:numpy;conv_2d:gemm;layers,optimizers:numpy,cython'."
                " Selection: More specific modules are attempted first, backend order goes from least to most priority."
                f" Backends: {', '.join(map(repr, backends))}."
                f" Alias: {', '.join(f'{key!r} = {value!r}' for key, value in Base._map_backend.items())}."
                f" Default: {ModelBase.backend!r}."
            ),
        )
        self.add_argument(
            "--batch-size",
            type=int,
            default=ModelBase.batch_size,
            help=(
                "Batch size per MPI rank."
                " Or 'batch_size' or 'global_batch_size' must have a value (but not both)."
                f" Default: {ModelBase.batch_size!r}."
            ),
        )
        self.add_argument(
            "--global-batch-size",
            type=int,
            default=ModelBase.global_batch_size,
            help=(
                "Batch size between all MPI ranks. "
                "Or 'batch_size' or 'global_batch_size' must have a value (but not both). "
                f"Default: {ModelBase.global_batch_size!r}."
            ),
        )
        dtype = list(map(np_dtype, ["float32", "float64"]))
        self.add_argument(
            "--dtype",
            type=np_dtype,
            default=ModelBase.dtype,
            choices=dtype,
            help=(
                f"Datatype to use: {', '.join(map(repr, map(str, dtype)))}."
                f" Default: {str(ModelBase.dtype)!r}."
            ),
        )
        self.add_argument(
            "--quantize",
            action=argparse.BooleanOptionalAction,
            default=ModelBase.quantize,
            help=(f"Enable model quantization. Default: {ModelBase.quantize!r}"),
        )
        quantize_dtype = list(map(np_dtype, ["float16", "float32", "float64"]))
        self.add_argument(
            "--quantize-dtype",
            type=np_dtype,
            default=ModelBase.quantize_dtype,
            choices=quantize_dtype,
            help=(
                f"Quantized datatype to use: {', '.join(map(repr, map(str, quantize_dtype)))}."
                f" Default: {str(ModelBase.quantize_dtype)!r}."
            ),
        )
        self.add_argument(
            "--num-epochs",
            type=int,
            default=ModelBase.num_epochs,
            help=(f"Number of epochs to perform. Default: {ModelBase.num_epochs!r}."),
        )
        self.add_argument(
            "--steps-per-epoch",
            type=int,
            default=ModelBase.steps_per_epoch,
            help=(
                "Trims the training data depending on the given number of steps per epoch. "
                "If '0', then no trim, full dataset."
                f"Default: {ModelBase.steps_per_epoch!r}."
            ),
        )
        self.add_argument(
            "--evaluate",
            dest="evaluate_on_train",
            action=argparse.BooleanOptionalAction,
            default=ModelBase.evaluate_on_train,
            help=(
                "Evaluate the model before and after training the model."
                f" Default: {ModelBase.evaluate_on_train!r}."
            ),
        )
        self.add_argument(
            "--evaluate-only",
            action=argparse.BooleanOptionalAction,
            default=ModelBase.evaluate_only,
            help=(f"Only evaluate the model. Default: {ModelBase.evaluate_only!r}."),
        )
        self.add_argument(
            "--model-state-filename",
            type=str,
            default=ModelBase.model_state_filename,
            help=(f"Load weights and bias from file. Default: {ModelBase.model_state_filename!r}."),
        )
        self.add_argument(
            "--history-file",
            type=str,
            default=ModelBase.history_file,
            help=(
                f"Filename to save training loss and metrics. Default: {ModelBase.history_file!r}."
            ),
        )
        self.add_argument(
            "--tensor-format",
            type=str,
            default=ModelBase.tensor_format,
            choices=TensorFormat,
            help=(
                f"Data format to be used: {', '.join(map(repr, map(str, TensorFormat)))}."
                f" If not defined value sets {str(TensorFormat.NCHW)!r} when cuDNN is available,"
                f" {str(TensorFormat.NHWC)!r} otherwise."
                f" Default: {ModelBase.tensor_format!r}."
            ),
        )
        self.add_argument(
            "--random-seed",
            type=int,
            default=ModelBase.random_seed,
            help=(f"Initial state of random number generator. Default: {ModelBase.random_seed!r}."),
        )
        self.add_argument(
            "--shared-tmp-memory",
            action=argparse.BooleanOptionalAction,
            default=ModelBase.shared_tmp_memory,
            help=(
                "Allows to use a memory pool for all the temporary data structures."
                f" Default: {ModelBase.shared_tmp_memory!r}."
            ),
        )

        # Synchronization options
        _sy_group = self.add_argument_group("Synchronization options")
        _sy_group.add_argument(
            "--shared-data",
            action=argparse.BooleanOptionalAction,
            default=ModelBase.shared_data,
            help=(
                "If 'True' ranks assume they share the file system."
                f"Default: {ModelBase.shared_data!r}."
            ),
        )
        _sy_group.add_argument(
            "--model-sync-freq",
            type=int,
            default=ModelBase.model_sync_freq,
            help=(
                "Number of batches between model synchronization."
                " The '0' value synchronizes gradients every batch."
                " Positive values synchronizes gradients and weights every N batches."
                " Negative values disables synchronization."
                f"Default: {ModelBase.model_sync_freq!r}."
            ),
        )
        _sy_group.add_argument(
            "--model-sync-algo",
            type=ModelBase.SyncAlgorithm,
            default=ModelBase.model_sync_algo,
            choices=ModelBase.SyncAlgorithm,
            help=(
                f"Aggregation method used to synchronize models: {', '.join(map(repr, map(str, ModelBase.SyncAlgorithm)))}."
                f" Default: {str(ModelBase.model_sync_algo)!r}."
            ),
        )
        _sy_group.add_argument(
            "--model-sync-participation",
            type=ModelBase.SyncParticipation,
            default=ModelBase.model_sync_participation,
            choices=ModelBase.SyncParticipation,
            help=(
                f"Rank participation to synchronize models: {', '.join(map(repr, map(str, ModelBase.SyncParticipation)))}."
                f" Default: {str(ModelBase.model_sync_participation)!r}."
            ),
        )
        _sy_group.add_argument(
            "--model-sync-min-avail",
            type=int,
            default=ModelBase.model_sync_min_avail,
            help=(
                "Minimum ranks with data required to synchronize models."
                f" Default: {ModelBase.model_sync_min_avail!r}."
            ),
        )
        _sy_group.add_argument(
            "--initial-model-sync",
            action=argparse.BooleanOptionalAction,
            default=ModelBase.initial_model_sync,
            help=(
                f"Synchronize models on training start. Default: {ModelBase.initial_model_sync!r}."
            ),
        )
        _sy_group.add_argument(
            "--final-model-sync",
            action=argparse.BooleanOptionalAction,
            default=ModelBase.final_model_sync,
            help=(f"Synchronize models on training end. Default: {ModelBase.final_model_sync!r}."),
        )
        _sy_group.add_argument(
            "--model-sync-quantize",
            action=argparse.BooleanOptionalAction,
            default=ModelBase.model_sync_quantize,
            help=(
                "Enable model quantization on synchronize."
                f" Default: {ModelBase.model_sync_quantize!r}."
            ),
        )
        model_sync_dtype = list(map(np_dtype, ["float16", "float32", "float64"]))
        _sy_group.add_argument(
            "--model-sync-dtype",
            type=np_dtype,
            default=ModelBase.model_sync_dtype,
            choices=model_sync_dtype,
            help=(
                f"Model synchronization quantization target dtype: {', '.join(map(repr, map(str, model_sync_dtype)))}"
                f" Default: {str(ModelBase.model_sync_dtype)!r}."
            ),
        )

        # Dataset options
        datasets = list_modules("datasets")
        datasets.remove("memory")
        _ds_group = self.add_argument_group("Dataset options")
        _ds_group.add_argument(
            "--dataset",
            dest="dataset_name",
            type=str,
            default="mnist",
            choices=datasets,
            help=(
                f"Dataset to train: {', '.join(map(repr, datasets[:3]))}, etc."
                f" Default: {ModelBase.dataset_name!r}."
            ),
        )
        _ds_group.add_argument(
            "--dataset-percentage",
            type=float,
            default=ModelBase.dataset_percentage,
            help=(
                "Percentage of dataset that will be used."
                " If it is '0': it is deactivated;"
                " if is is a value below '1' (and above 0): it will perform undersampling;"
                " and if is is a value above '1': it will perform oversampling."
                f" Default: {ModelBase.dataset_percentage!r}."
            ),
        )
        _ds_group.add_argument(
            "--dataset-path",
            type=str,
            default=ModelBase.dataset_path,
            help=(f"Path to the dataset. Default: {ModelBase.dataset_path!r}."),
        )
        _ds_group.add_argument(
            "--dataset-lang",
            type=str,
            default=ModelBase.dataset_lang,
            help=(f"Dataset language. Default: {ModelBase.dataset_lang!r}."),
        )
        _ds_group.add_argument(
            "--dataset-lang2",
            type=str,
            default=ModelBase.dataset_lang2,
            help=(f"Dataset second language. Default: {ModelBase.dataset_lang2!r}."),
        )
        _ds_group.add_argument(
            "--synthetic-train-samples",
            type=int,
            default=ModelBase.synthetic_train_samples,
            help=(
                f"Number of synthetic train sample. Default: {ModelBase.synthetic_train_samples!r}."
            ),
        )
        _ds_group.add_argument(
            "--synthetic-test-samples",
            type=int,
            default=ModelBase.synthetic_test_samples,
            help=(
                f"Number of synthetic train sample. Default: {ModelBase.synthetic_test_samples!r}."
            ),
        )
        _ds_group.add_argument(
            "--synthetic-input-shape",
            type=csi,
            default=ModelBase.synthetic_input_shape,
            help=(
                "Synthetic input shape (coma separated)."
                f" Default: {ModelBase.synthetic_input_shape!r}."
            ),
        )
        _ds_group.add_argument(
            "--synthetic-output-shape",
            type=csi,
            default=ModelBase.synthetic_output_shape,
            help=(
                "Synthetic output shape (coma separated)."
                f" Default: {ModelBase.synthetic_output_shape!r}."
            ),
        )
        _ds_group.add_argument(
            "--test-as-validation",
            action=argparse.BooleanOptionalAction,
            default=ModelBase.test_as_validation,
            help=(
                "Prevent making partitions on training data for training+validation data,"
                " use test data for validation. True if specified."
                f" Default: {ModelBase.test_as_validation!r}."
            ),
        )
        _ds_group.add_argument(
            "--validation-split",
            type=factor,
            default=ModelBase.validation_split,
            help=(
                "Split between training and validation data."
                f" Default: {ModelBase.validation_split!r}."
            ),
        )
        _ds_group.add_argument(
            "--augment-shuffle",
            action=argparse.BooleanOptionalAction,
            default=ModelBase.augment_shuffle,
            help=(f"Shuffle training images. Default: {ModelBase.augment_shuffle!r}."),
        )
        _ds_group.add_argument(
            "--augment-horizontal-flip",
            type=factor,
            default=ModelBase.augment_horizontal_flip,
            help=(
                "Probability to do a horizontal flip to the training images."
                " If the value is less or equal to 0 it is disabled."
                f" Default: {ModelBase.augment_horizontal_flip!r}."
            ),
        )
        _ds_group.add_argument(
            "--augment-vertical-flip",
            type=factor,
            default=ModelBase.augment_vertical_flip,
            help=(
                "Probability to do a vertical flip to the training images."
                " If the value is less or equal to 0 it is disabled."
                f" Default: {ModelBase.augment_vertical_flip!r}."
            ),
        )
        _ds_group.add_argument(
            "--augment-rotate",
            type=factor,
            default=ModelBase.augment_rotate,
            help=(
                "Probability to rotate training images."
                " If the value is less or equal to 0 it is disabled."
                f" Default: {ModelBase.augment_rotate!r}."
            ),
        )
        _ds_group.add_argument(
            "--augment-rotate-degree",
            type=float,
            default=ModelBase.augment_rotate_degree,
            help=(
                "The maximum degree to rotate training images."
                f" Default: {ModelBase.augment_rotate_degree!r}."
            ),
        )
        _ds_group.add_argument(
            "--augment-brightness",
            type=factor,
            default=ModelBase.augment_brightness,
            help=(
                "Probability to change the brightness to training images."
                " If the value is less or equal to 0 it is disabled."
                f" Default: {ModelBase.augment_brightness!r}."
            ),
        )
        _ds_group.add_argument(
            "--augment-brightness-factor",
            type=float,
            default=ModelBase.augment_brightness_factor,
            help=(
                "The maximum brightness to apply in training images."
                " Value ranges from 0 (no brightness), to 1 (same), up to infinity."
                f" Default: {ModelBase.augment_brightness_factor!r}."
            ),
        )
        _ds_group.add_argument(
            "--augment-contrast",
            type=factor,
            default=ModelBase.augment_contrast,
            help=(
                "Probability to change the contrast to training images."
                " If the value is less or equal to 0 it is disabled."
                f" Default: {ModelBase.augment_contrast!r}."
            ),
        )
        _ds_group.add_argument(
            "--augment-contrast-factor",
            type=float,
            default=ModelBase.augment_contrast_factor,
            help=(
                "The maximum contrast to apply in training images."
                " Value ranges from 0 (no brightness), to 1 (same), up to infinity."
                f" Default: {ModelBase.augment_contrast_factor!r}."
            ),
        )
        _ds_group.add_argument(
            "--augment-saturation",
            type=factor,
            default=ModelBase.augment_saturation,
            help=(
                "Probability to change the saturation to training images."
                " If the value is less or equal to 0 it is disabled."
                f" Default: {ModelBase.augment_saturation!r}."
            ),
        )
        _ds_group.add_argument(
            "--augment-saturation-factor",
            type=float,
            default=ModelBase.augment_saturation_factor,
            help=(
                "The maximum saturation to apply in training images."
                " Value ranges from 0 (no brightness), to 1 (same), up to infinity."
                f" Default: {ModelBase.augment_saturation_factor!r}."
            ),
        )
        _ds_group.add_argument(
            "--augment-mask",
            type=factor,
            default=ModelBase.augment_mask,
            help=(
                "Probability to mask training images."
                " If the value is less or equal to 0 it is disabled."
                f" Default: {ModelBase.augment_mask!r}."
            ),
        )
        _ds_group.add_argument(
            "--augment-mask-size",
            type=int,
            default=ModelBase.augment_mask_size,
            help=(f"Size to mask training images. Default: {ModelBase.augment_mask_size!r}."),
        )
        _ds_group.add_argument(
            "--augment-blur",
            type=factor,
            default=ModelBase.augment_blur,
            help=(
                "Probability to blur training images."
                " If the value is less or equal to 0 it is disabled."
                f" Default: {ModelBase.augment_blur!r}."
            ),
        )
        _ds_group.add_argument(
            "--augment-blur-size",
            type=int,
            default=ModelBase.augment_blur_size,
            help=(f"Size to blur training images. Default: {ModelBase.augment_blur_size!r}."),
        )
        _ds_group.add_argument(
            "--augment-crop",
            action=argparse.BooleanOptionalAction,
            default=ModelBase.augment_crop,
            help=(f"Crop the images. True if specified. Default: {ModelBase.augment_crop!r}."),
        )
        _ds_group.add_argument(
            "--augment-crop-perc",
            type=factor,
            default=ModelBase.augment_crop_perc,
            help=(
                f"Central crop percentage of the images. Default: {ModelBase.augment_crop_perc!r}."
            ),
        )
        _ds_group.add_argument(
            "--augment-scale",
            action=argparse.BooleanOptionalAction,
            default=ModelBase.augment_scale,
            help=(f"Resize the images. True if specified. Default: {ModelBase.augment_scale!r}."),
        )
        _ds_group.add_argument(
            "--augment-scale-size",
            type=int,
            default=ModelBase.augment_scale_size,
            help=(f"New size of the images. Default: {ModelBase.augment_scale_size!r}."),
        )
        _ds_group.add_argument(
            "--augment-perspective",
            type=factor,
            default=ModelBase.augment_perspective,
            help=(
                "Probability to change the perspective in training images."
                " If the value is less or equal to 0 it is disabled."
                f" Default: {ModelBase.augment_perspective!r}."
            ),
        )
        _ds_group.add_argument(
            "--augment-perspective-factor",
            type=factor,
            default=ModelBase.augment_perspective_factor,
            help=(
                "The perspective distortion factor. The ranges are from 0.0 to 0.5."
                f" Default: {ModelBase.augment_perspective_factor!r}."
            ),
        )
        _ds_group.add_argument(
            "--augment-normalize",
            action=argparse.BooleanOptionalAction,
            default=ModelBase.augment_normalize,
            help=(f"Normalize dataset. Default: {ModelBase.augment_normalize!r}."),
        )
        _ds_group.add_argument(
            "--augment-normalize-offset",
            type=float,
            default=ModelBase.augment_normalize_offset,
            help=(f"Offset samples by a value. Default: {ModelBase.augment_normalize_offset!r}."),
        )
        _ds_group.add_argument(
            "--augment-normalize-scale",
            type=float,
            default=ModelBase.augment_normalize_scale,
            help=(f"Scale samples by a value. Default: {ModelBase.augment_normalize_scale!r}."),
        )

        # Optimization options
        _oo_group = self.add_argument_group("Optimization options")
        _oo_group.add_argument(
            "--fused-bn-relu",
            action=argparse.BooleanOptionalAction,
            default=ModelBase.fused_bn_relu,
            help=(
                "Fuse BatchNormalization and Relu layers. True if specified."
                f" Default: {ModelBase.fused_bn_relu!r}."
            ),
        )
        _oo_group.add_argument(
            "--fused-conv-relu",
            action=argparse.BooleanOptionalAction,
            default=ModelBase.fused_conv_relu,
            help=(
                "Fuse Conv2D and Relu layers. True if specified."
                f" Default: {ModelBase.fused_conv_relu!r}."
            ),
        )
        _oo_group.add_argument(
            "--fused-conv-bn",
            action=argparse.BooleanOptionalAction,
            default=ModelBase.fused_conv_bn,
            help=(
                "Fuse Conv2D and BatchNormalization layers. True if specified."
                f" Default: {ModelBase.fused_conv_bn!r}."
            ),
        )
        _oo_group.add_argument(
            "--fused-conv-bn-relu",
            action=argparse.BooleanOptionalAction,
            default=ModelBase.fused_conv_bn_relu,
            help=(
                "Fuse Conv2D and BatchNormalization and Relu layers. True if specified."
                f" Default: {ModelBase.fused_conv_bn_relu!r}."
            ),
        )

        # Convolution methods
        _cm_group = self.add_argument_group("Convolution options")
        _cm_group.add_argument(
            "--conv-direct-method",
            type=str,
            default=ModelBase.conv_direct_method,
            help=(
                "ConvDirect algorithm to use in Conv2D layers."
                " Use 'convDirect_info' to see available algorithms."
                " Default: 'convdirect_original_{tensor_format}_default'"
            ),
        )

        # Optimizer options
        optimizers = list_modules("optimizers")
        _op_group = self.add_argument_group("Optimizer options")
        _op_group.add_argument(
            "--optimizer",
            dest="optimizer_name",
            type=str,
            default=ModelBase.optimizer_name,
            choices=optimizers,
            help=(
                f"Optimizers: {', '.join(map(repr, optimizers[:3]))}, etc."
                f" Default: {ModelBase.optimizer_name!r}."
            ),
        )
        _op_group.add_argument(
            "--learning-rate",
            type=float,
            default=ModelBase.learning_rate,
            help=(f"Learning rate. Default: {ModelBase.learning_rate!r}."),
        )
        _op_group.add_argument(
            "--learning-rate-scaling",
            action=argparse.BooleanOptionalAction,
            default=ModelBase.learning_rate_scaling,
            help=(
                "Scale learning rate in parallelism: new_lr = lr * num_procs. True if specified."
                " If left undefined, when '--batch-size' is defined, defaults to True."
                f" Default: {ModelBase.learning_rate_scaling!r}."
            ),
        )
        _op_group.add_argument(
            "--optimizer-momentum",
            type=float,
            default=ModelBase.optimizer_momentum,
            help=(f"Decay rate for optimizers. Default: {ModelBase.optimizer_momentum!r}."),
        )
        _op_group.add_argument(
            "--optimizer-decay",
            type=float,
            default=ModelBase.optimizer_decay,
            help=(f"Decay rate for optimizers. Default: {ModelBase.optimizer_decay!r}."),
        )
        _op_group.add_argument(
            "--optimizer-nesterov",
            action=argparse.BooleanOptionalAction,
            default=ModelBase.optimizer_nesterov,
            help=(
                f"Whether to apply Nesterov momentum. Default: {ModelBase.optimizer_nesterov!r}."
            ),
        )
        _op_group.add_argument(
            "--optimizer-beta1",
            type=float,
            default=ModelBase.optimizer_beta1,
            help=(
                f"Variable for 'adam', 'nadam' optimizers. Default: {ModelBase.optimizer_beta1!r}."
            ),
        )
        _op_group.add_argument(
            "--optimizer-beta2",
            type=float,
            default=ModelBase.optimizer_beta2,
            help=(
                f"Variable for 'adam', 'nadam' optimizers. Default: {ModelBase.optimizer_beta2!r}."
            ),
        )
        _op_group.add_argument(
            "--optimizer-epsilon",
            type=float,
            default=ModelBase.optimizer_epsilon,
            help=(
                "Variable for 'rmsprop', 'adam', 'nadam'."
                f" Default: {ModelBase.optimizer_epsilon!r}."
            ),
        )
        _op_group.add_argument(
            "--optimizer-rho",
            type=float,
            default=ModelBase.optimizer_rho,
            help=(f"Variable for 'rmsprop' optimizers. Default: {ModelBase.optimizer_rho!r}."),
        )
        _op_group.add_argument(
            "--optimizer-tau",
            type=int,
            default=ModelBase.optimizer_tau,
            help=(f"Variable for 'oktopk' optimizers. Default: {ModelBase.optimizer_tau!r}."),
        )
        _op_group.add_argument(
            "--optimizer-tau-prime",
            type=int,
            default=ModelBase.optimizer_tau,
            help=(f"Variable for 'oktopk' optimizers. Default: {ModelBase.optimizer_tau!r}."),
        )
        _op_group.add_argument(
            "--optimizer-density",
            type=float,
            default=ModelBase.optimizer_density,
            help=(f"Variable for 'oktopk' optimizers. Default: {ModelBase.optimizer_density!r}."),
        )
        _op_group.add_argument(
            "--oktopk-min-k",
            type=int,
            default=ModelBase.oktopk_min_k,
            help=(f"Variable for 'oktopk' optimizers. Default: {ModelBase.oktopk_min_k!r}."),
        )
        losses = list_modules("losses")
        _op_group.add_argument(
            "--loss-func",
            dest="loss_func_name",
            type=str,
            default=ModelBase.loss_func_name,
            choices=losses,
            help=(
                "Loss functions that is evaluated on each trained batch:"
                f" {', '.join(map(repr, losses[:3]))}, etc."
                f" Default: {ModelBase.loss_func_name!r}."
            ),
        )
        _op_group.add_argument(
            "--loss-eps",
            type=float,
            default=ModelBase.loss_eps,
            help=(f"Value for numerical stability. Default: {ModelBase.loss_eps!r}."),
        )
        _op_group.add_argument(
            "--loss-weights",
            type=csf,
            default=ModelBase.loss_weights,
            help=(
                "List modifiers separated by a comma to indicate the weights of every class."
                " If the value is 'None' it will use the default dataset's value; "
                " if the dataset has not a default value, all classes will weight '1'."
                " Example, with 3 classes: '0.4,1.8,0.2'."
                f" Default: {ModelBase.loss_weights!r}."
            ),
        )
        _op_group.add_argument(
            "--use-loss-weights",
            action=argparse.BooleanOptionalAction,
            default=ModelBase.use_loss_weights,
            help=(
                "True if use the loss-weights parameter, "
                " False to set all classes' weights with the same value."
                f" Default: {ModelBase.use_loss_weights!r}."
            ),
        )
        metrics = list_modules("metrics")
        _op_group.add_argument(
            "--metrics",
            type=css,
            default=ModelBase.metrics,
            help=(
                "List of comma-separated metrics that are evaluated on each trained batch:"
                f" {', '.join(map(repr, metrics))}, etc."
                f" Default: {ModelBase.metrics!r}."
            ),
        )

        # Schedulers options
        scheduler_metric = []
        for metric in (*list_modules("losses"), *list_modules("metrics")):
            for part in DatasetBase.Part:
                scheduler_metric.append(f"{part._name_.lower()}_{metric}")
        schedulers = list_modules("schedulers")
        _sh_group = self.add_argument_group("Schedulers options")
        _sh_group.add_argument(
            "--schedulers",
            dest="schedulers_names",
            type=css,
            default=ModelBase.schedulers_names,
            help=(
                "List of comma-separated schedulers:"
                f" {', '.join(map(repr, schedulers))}, etc."
                f" Default: {ModelBase.schedulers_names!r}."
            ),
        )
        _sh_group.add_argument(
            "--warm-up-epochs",
            type=int,
            default=ModelBase.warm_up_epochs,
            help=(
                "Number of batches (ramp up) that the LR is scaled up from 0 until LR."
                f" Default: {ModelBase.warm_up_epochs!r}."
            ),
        )
        _sh_group.add_argument(
            "--early-stopping-metric",
            type=str,
            default=ModelBase.early_stopping_metric,
            choices=scheduler_metric,
            help=(
                "Loss metric monitored by early_stopping LR scheduler."
                f" Default: {ModelBase.early_stopping_metric!r}."
            ),
        )
        _sh_group.add_argument(
            "--early-stopping-patience",
            type=int,
            default=ModelBase.early_stopping_patience,
            help=(
                "Number of epochs with no improvement after which training will be stopped."
                f" Default: {ModelBase.early_stopping_patience!r}."
            ),
        )
        _sh_group.add_argument(
            "--early-stopping-minimize",
            action=argparse.BooleanOptionalAction,
            default=ModelBase.early_stopping_minimize,
            help=(
                "Whether to minimize the metric. If False, it will maximize."
                f" Default: {ModelBase.early_stopping_minimize!r}."
            ),
        )
        _sh_group.add_argument(
            "--reduce-lr-on-plateau-metric",
            type=str,
            default=ModelBase.reduce_lr_on_plateau_metric,
            choices=scheduler_metric,
            help=(
                "Loss metric monitored by reduce_lr_on_plateau LR scheduler."
                f" Default: {ModelBase.reduce_lr_on_plateau_metric!r}."
            ),
        )
        _sh_group.add_argument(
            "--reduce-lr-on-plateau-factor",
            type=float,
            default=ModelBase.reduce_lr_on_plateau_factor,
            help=(
                "Factor by which the learning rate will be reduced. new_lr = lr * factor."
                f" Default: {ModelBase.reduce_lr_on_plateau_factor!r}."
            ),
        )
        _sh_group.add_argument(
            "--reduce-lr-on-plateau-patience",
            type=int,
            default=ModelBase.reduce_lr_on_plateau_patience,
            help=(
                "Number of epochs with no improvement after which LR will be reduced."
                f" Default: {ModelBase.reduce_lr_on_plateau_patience!r}."
            ),
        )
        _sh_group.add_argument(
            "--reduce-lr-on-plateau-min-lr",
            type=float,
            default=ModelBase.reduce_lr_every_nepochs_min_lr,
            help=(
                "Lower bound on the learning rate."
                f" Default: {ModelBase.reduce_lr_every_nepochs_min_lr!r}."
            ),
        )
        _sh_group.add_argument(
            "--reduce-lr-every-nepochs-factor",
            type=float,
            default=ModelBase.reduce_lr_every_nepochs_factor,
            help=(
                "Factor by which the learning rate will be reduced. new_lr = lr * factor."
                f" Default: {ModelBase.reduce_lr_every_nepochs_factor!r}."
            ),
        )
        _sh_group.add_argument(
            "--reduce-lr-every-nepochs-nepochs",
            type=int,
            default=ModelBase.reduce_lr_every_nepochs_nepochs,
            help=(
                "Number of epochs after which LR will be periodically reduced."
                f" Default: {ModelBase.reduce_lr_every_nepochs_nepochs!r}."
            ),
        )
        _sh_group.add_argument(
            "--reduce-lr-every-nepochs-min-lr",
            type=float,
            default=ModelBase.reduce_lr_every_nepochs_min_lr,
            help=(
                "Lower bound on the learning rate."
                f" Default: {ModelBase.reduce_lr_every_nepochs_min_lr!r}."
            ),
        )
        _sh_group.add_argument(
            "--stop-at-loss-metric",
            type=str,
            default=ModelBase.stop_at_loss_metric,
            choices=scheduler_metric,
            help=(
                "Loss metric monitored by stop_at_loss LR scheduler."
                f" Default: {ModelBase.stop_at_loss_metric!r}."
            ),
        )
        _sh_group.add_argument(
            "--stop-at-loss-threshold",
            type=float,
            default=ModelBase.stop_at_loss_threshold,
            help=(
                "Metric threshold monitored by stop_at_loss LR scheduler."
                f" Default: {ModelBase.stop_at_loss_threshold!r}."
            ),
        )
        _sh_group.add_argument(
            "--model-checkpoint-metric",
            type=str,
            default=ModelBase.model_checkpoint_metric,
            choices=scheduler_metric,
            help=(
                "Loss metric monitored by model_checkpoint LR scheduler."
                f" Default: {ModelBase.model_checkpoint_metric!r}."
            ),
        )
        _sh_group.add_argument(
            "--model-checkpoint-save-freq",
            type=int,
            default=ModelBase.model_checkpoint_save_freq,
            help=(
                "Frequency (in epochs) at which the model weights and bias"
                " will be saved by the model_checkpoint LR scheduler."
                f" Default: {ModelBase.model_checkpoint_save_freq!r}."
            ),
        )

        # Parallel execution options
        _pe_group = self.add_argument_group("Parallel execution options")
        _pe_group.add_argument(
            "--parallel-data",
            action=argparse.BooleanOptionalAction,
            default=ModelBase.parallel_data,
            help=(f"Enable data parallelization modes. Default: {ModelBase.parallel_data!r}."),
        )
        _pe_group.add_argument(
            "--parallel-pipeline",
            action=argparse.BooleanOptionalAction,
            default=ModelBase.parallel_pipeline,
            help=(
                f"Enable pipeline parallelization modes. Default: {ModelBase.parallel_pipeline!r}."
            ),
        )
        _pe_group.add_argument(
            "--use-blocking-mpi",
            action=argparse.BooleanOptionalAction,
            default=ModelBase.use_blocking_mpi,
            help=(f"Enable non-blocking MPI primitives. Default: {ModelBase.use_blocking_mpi!r}."),
        )
        _pe_group.add_argument(
            "--use-mpi-buffers",
            action=argparse.BooleanOptionalAction,
            default=ModelBase.use_mpi_buffers,
            help=(
                "Enable the use of MPI buffers. Possible values: 'True' (MPI operations by buffer),"
                " 'False' (MPI operations by object) or undefined (auto-select the better option)."
                f" Default: {ModelBase.use_mpi_buffers!r}."
            ),
        )
        _pe_group.add_argument(
            "--use-cudnn",
            action=argparse.BooleanOptionalAction,
            default=ModelBase.use_cudnn,
            help=(
                "Ignored, always enabled if plausible, present just for compatibility."
                f" Default: {ModelBase.use_cudnn!r}."
            ),
        )
        _pe_group.add_argument(
            "--use-gpudirect",
            action=argparse.BooleanOptionalAction,
            default=ModelBase.use_gpudirect,
            help=(
                "Enable GPU pinned memory for gradients when using a CUDA-aware MPI version."
                f" Default: {ModelBase.use_gpudirect!r}."
            ),
        )
        _pe_group.add_argument(
            "--use-nccl",
            action=argparse.BooleanOptionalAction,
            default=ModelBase.use_nccl,
            help=(
                "Enable the use of the NCCL library for collective communications on GPUs."
                " This option can only be set when cuDNN is available."
                f" Default: {ModelBase.use_nccl!r}."
            ),
        )
        _pe_group.add_argument(
            "--use-cudnn-auto-conv-algo",
            action=argparse.BooleanOptionalAction,
            default=ModelBase.use_cudnn_auto_conv_algo,
            help=(
                "Let cuDNN to select the best performing convolution algorithm."
                f" Default: {ModelBase.use_cudnn_auto_conv_algo!r}."
            ),
            deprecated=True,
        )

        # Encryption options
        _cy_group = self.add_argument_group("Encryption options")
        _cy_group.add_argument(
            "--encryption",
            dest="encryption_name",
            type=str,
            default=ModelBase.encryption_name,
            help=(
                "Encryption library backend to use."
                " Use 'polyhe.Backend' to see available libraries."
                f" Default: {ModelBase.encryption_name!r}."
            ),
        )
        _cy_group.add_argument(
            "--encryption-slots",
            type=int,
            default=ModelBase.encryption_slots,
            help=(f"Encryption slot count. 2 ^ 'value'. Default: {ModelBase.encryption_slots!r}."),
        )
        _cy_group.add_argument(
            "--encryption-scale",
            type=int,
            default=ModelBase.encryption_scale,
            help=(
                "Encryption operational scale. 2 ^ 'value'."
                f" Default: {ModelBase.encryption_scale!r}."
            ),
        )
        _cy_group.add_argument(
            "--encryption-security",
            type=int,
            default=ModelBase.encryption_security,
            help=(
                "Encryption security level: 128, 192, 256."
                " Use 'polyhe.{backend}.SECURITY_LEVEL' to see available security levels."
                f" Default: {ModelBase.encryption_security!r}."
            ),
        )

        # Tracing and profiling
        _tr_group = self.add_argument_group("Tracing options")
        _tr_group.add_argument(
            "--tracing",
            action=argparse.BooleanOptionalAction,
            default=ModelBase.tracing,
            help=(f"Obtain Simple/Extrae-based traces. Default: {ModelBase.tracing!r}."),
        )
        _tr_group.add_argument(
            "--tracer-output",
            type=str,
            default=ModelBase.tracer_output,
            help=(
                "Output file to store the Simple/Extrae-based traces."
                f" Default: {ModelBase.tracer_output!r}."
            ),
        )
        _tr_group.add_argument(
            "--tracer-pmlib-server",
            type=str,
            default=ModelBase.tracer_pmlib_server,
            help=(f"Address of PMlib tracer server. Default: {ModelBase.tracer_pmlib_server!r}."),
        )
        _tr_group.add_argument(
            "--tracer-pmlib-port",
            type=int,
            default=ModelBase.tracer_pmlib_port,
            help=(f"Port of PMlib tracer server. Default: {ModelBase.tracer_pmlib_port!r}."),
        )
        _tr_group.add_argument(
            "--tracer-pmlib-device",
            type=str,
            default=ModelBase.tracer_pmlib_device,
            help=(f"Port of PMlib tracer device. Default: {ModelBase.tracer_pmlib_device!r}."),
        )
        _tr_group.add_argument(
            "--profile",
            action=argparse.BooleanOptionalAction,
            default=ModelBase.profile,
            help=(f"Obtain Python profiles. Default: {ModelBase.profile!r}."),
        )

        # Performance modeling options
        _pm_group = self.add_argument_group("Performance modeling options")
        _pm_group.add_argument(
            "--cpu-speed", type=float, default=ModelBase.cpu_speed, help=argparse.SUPPRESS
        )
        _pm_group.add_argument(
            "--memory-bw", type=float, default=ModelBase.memory_bw, help=argparse.SUPPRESS
        )
        _pm_group.add_argument(
            "--network-bw", type=float, default=ModelBase.network_bw, help=argparse.SUPPRESS
        )
        _pm_group.add_argument(
            "--network-lat", type=float, default=ModelBase.network_lat, help=argparse.SUPPRESS
        )
        _pm_group.add_argument(
            "--network-algo",
            type=NetworkAlgoEnum,
            default=ModelBase.network_algo,
            choices=NetworkAlgoEnum,
            help=argparse.SUPPRESS,
        )

        # Add Runtime parallel execution options
        _re_group = self.add_argument_group("Runtime parallel execution options")
        _re_group.add_argument("--gpus-per-node", type=int, default=-1, help=argparse.SUPPRESS)
        _re_group.add_argument("--mpi-processes", type=int, default=-1, help=argparse.SUPPRESS)
        _re_group.add_argument(
            "--threads-per-process", type=int, default=-1, help=argparse.SUPPRESS
        )

        # Add Communication options
        _cm_group = self.add_argument_group("Communication options")
        _cm_group.add_argument("--mpi-protocol", type=str, default="", help=argparse.SUPPRESS)
        _cm_group.add_argument("--mpi-server", type=str, default="", help=argparse.SUPPRESS)
        _cm_group.add_argument("--mpi-port", type=int, default=-1, help=argparse.SUPPRESS)

    def parse_args(self, args: Sequence[str] | None = None) -> Namespace:
        """Parses command line arguments and injects runtime environment data."""
        # Call super.parse_args
        namespace = Namespace()
        result = super().parse_args(args, namespace)
        # Add runtime data
        result.mpi_processes = _get_mpi_processes()
        result.threads_per_process = _get_threads_per_process()
        result.gpus_per_node = get_gpus_per_node()
        result.mpi_protocol = _get_mpi_protocol()
        result.mpi_server = _get_mpi_server()
        result.mpi_port = _get_mpi_port()
        result.use_cudnn = _get_use_cudnn()
        result.groups = self._action_groups
        return result

    def __str__(self) -> str:
        """
        Converts ArgumentParser into a Markdown document.

        The output is intentionally very similar to argparse --help,
        only adding Markdown formatting.
        """

        lines = []

        def quotes(text: str) -> str:
            return re.sub(r"'([^']*)'", r"`\1`", text).replace("``", "` `")

        for group in self._action_groups:
            actions = [a for a in group._group_actions if a.help is not argparse.SUPPRESS]

            if not actions:
                continue

            description = quotes(group.description or "")
            lines.append(f"- {group.title}: {description}".rstrip())

            for action in actions:
                if not action.option_strings:
                    option = f"`{action.dest}`"
                else:
                    option = ", ".join(f"`{opt}`" for opt in action.option_strings)

                description = quotes(action.help or "")
                description = textwrap.fill(
                    description,
                    width=80,
                    initial_indent="",
                    subsequent_indent="",
                ).replace("\n", "\n    ")
                lines.append(f"  - {option}: {description}".rstrip())

            lines.append("")

        return "\n".join(lines)


if __name__ == "__main__":
    print(ArgumentParser())
