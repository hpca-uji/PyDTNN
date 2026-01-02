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
import multiprocessing
import os

import numpy as np
from pydtnn.utils import parse_bool as bool_lambda
from functools import cache

from typing import Any


def factor(x):
    """Returns x, which must be 0.0 < x <= 1.0"""
    x = float(x)
    if not (0.0 < x <= 1.0):
        raise ValueError("Provided value must be greater than 0.0 and less or equal to 1.0")
    return x


def np_dtype(x):
    """Returns a numpy object from a string representing the data type"""
    return getattr(np, x)


_this_file_path = os.path.dirname(os.path.realpath(__file__))
_scripts_path = os.path.join(_this_file_path, "scripts")
_default_dataset_path = os.path.join(_this_file_path, "datasets/mnist")
_desc = "Trains or evaluates a neural network using PyDTNN."
_epilogue = f"""Example scripts that call this program for training
and evaluating different neural network models with different datasets are
available at: '{_scripts_path}'."""


def _get_mpi_processes():
    try:
        # noinspection PyUnresolvedReferences,PyPackageRequirements
        from pympi import MPI
    except Exception:
        mpi_processes = 1
    else:
        mpi_processes = MPI.COMM_WORLD.Get_size()
    return mpi_processes


def _get_threads_per_process():
    #  From IBM OpenMP documentation: If you do not set OMP_NUM_THREADS, the number of processors available is the
    #  default value to form a new team for the first encountered parallel construct.
    threads_per_process = os.environ.get("OMP_NUM_THREADS", multiprocessing.cpu_count())
    return threads_per_process


def _get_gpus_per_node():
    import subprocess
    try:
        gpus_per_node = subprocess.check_output(["nvidia-smi", "-L"]).count(b'UUID')
    except (FileNotFoundError, subprocess.CalledProcessError):
        gpus_per_node = 0
    return gpus_per_node


def _get_mpi_protocol():
    try:
        from pydtnn.libs.libmpi import proto as PROTOCOL
        from pydtnn.libs.libmpi import ssl as SSL
    except Exception as e:
        PROTOCOL = None
        SSL = None
    protocol = str(PROTOCOL)
    if PROTOCOL and SSL:
        protocol = f"{protocol}+tls"
    return protocol


def _get_mpi_server():
    try:
        from pydtnn.libs.libmpi import addr
    except Exception as e:
        addr = None
    return addr


def _get_mpi_port():
    try:
        from pydtnn.libs.libmpi import port
    except Exception as e:
        port = None
    return port

# NOTE: with @cache it's not possible to extend the class.


@cache  # <== Singleton
class PydtnnArgumentParser(argparse.ArgumentParser):
    lines = []

    def __init__(self):
        super().__init__(description=_desc, epilog=_epilogue)
        # Parser and the supported arguments with their default values
        # (argparse.SUPPRESS is used to avoid showing them on the message)

        # Model
        self.add_argument('--model', dest="model_name", type=str, default=None,
                          help="Neural network model: \'simplemlp\', \'simplecnn\', \'alexnet\', \'vgg11\', \'vgg16\', etc. Default: \'None\'.")
        self.add_argument('--batch-size', type=int, default=None,
                          help="Batch size per MPI rank. Or \'batch_size\' or \'global_batch_size\' must have a value different from \'None\' (but not both). Default: \'None\'.")
        self.add_argument('--global-batch-size', type=int, default=None,
                          help="Batch size between all MPI ranks. Or \'batch_size\' or \'global_batch_size\' must have a value different from \'None\' (but not both) Default: \'None\'.")
        self.add_argument('--dtype', type=np_dtype, default=np.float32,
                          help="Datatype to use: \'float32\', \'float64\'. Default: float32.")
        self.add_argument('--num-epochs', type=int, default=1,
                          help="Number of epochs to perform. Default: 1.")
        self.add_argument('--steps-per-epoch', type=float, default=0,
                          help="Trims the training data depending on the given number of steps per epoch. Default: 0, i.e., do not trim.")
        self.add_argument('--evaluate', dest="evaluate_on_train", default=False, type=bool_lambda,
                          help="Evaluate the model before and after training the model. Default: False.")
        self.add_argument('--evaluate-only', default=False, type=bool_lambda,
                          help="Only evaluate the model. Default: False.")
        self.add_argument('--weights-and-bias-filename', type=str, default=None,
                          help="Load weights and bias from file. Default: None.")
        self.add_argument('--history-file', type=str, default=None,
                          help="Filename to save training loss and metrics.")
        self.add_argument('--tensor-format', type=lambda s: str(s).upper(), default="NHWC",
                          help="Data format to be used: \'NHWC\' or \'NCHW\'. Optionally, the \'AUTO\' value sets \'NCHW\' when the option \'--enable-gpu\' is set and \'NHWC\' otherwise. Default: \'NHWC\'.")
        self.add_argument('--random-seed', type=int, default=57005,
                          help='Initial state of random number generator. Default: 57005.')

        # Synchronization options
        _sy_group = self.add_argument_group("Synchronization options")
        _sy_group.add_argument('--shared-storage', default=True, type=bool_lambda,
                               help="If \'True\' ranks assume they share the file system. Default: True.")
        _sy_group.add_argument('--model-sync-freq', type=int, default=0,
                               help="Number of batches between model syncronization. The \'0\' value syncronizes gradients every batch. Positive values syncronizes gradients and weights every N batches. Negative values disables syncronization. Default: 0.")
        _sy_group.add_argument('--model-sync-alg', type=str, default="avg", choices=["avg", "wavg", "invwavg"],
                               help="Aggregation method used to syncronize models: \'avg\', \'wavg\' or \'invwavg\'. Default: \'avg\'.")
        _sy_group.add_argument('--model-sync-participation', type=str, default="all", choices=["all", "avail2all"],
                               help="Rank participation to syncronize models: \'all\' or \'avail2all\'. Default: \'all\'.")
        _sy_group.add_argument('--model-sync-min-avail', type=int, default=0,
                               help="Minumun ranks with data required to syncronize models. Default: 0.")
        _sy_group.add_argument('--initial-model-sync', type=bool_lambda, default=True,
                               help="Sincronize models on training start. Default: True.")
        _sy_group.add_argument('--final-model-sync', type=bool_lambda, default=True,
                               help="Sincronize models on training end. Default: True.")

        # Dataset options
        _ds_group = self.add_argument_group("Dataset options")
        _ds_group.add_argument('--dataset', dest="dataset_name", type=str, default=None, choices=["mnist", "cifar10", "imagenet", "archive", "folder", "chestxray", "synthetic"],
                               help="Dataset to train: \'mnist\', \'cifar10\', \'cyclone\', \'tsunamis\', \'imagenet\', \'archive\', \'folder\' \'chestxray\' or \'synthetic\'. Default: \'None\'.")
        _ds_group.add_argument('--dataset-percentage', type=float, default=0.0,
                               help="Percentage of dataset that will be used. If it is \'0\': it is deactivated; if is is a value below \'1\' (and above 0): it will perform undersampling; and if is is a value above \'1\': it will perform oversampling. Default: 0.")
        _ds_group.add_argument('--dataset-path', type=str, default=_default_dataset_path,
                               help="Path to the dataset.")
        _ds_group.add_argument('--dataset-lang', type=str, default="en",
                               help="Dataset language.")
        _ds_group.add_argument('--dataset-lang2', type=str, default="de",
                               help="Dataset second language.")
        _ds_group.add_argument('--synthetic-train-samples', default=1000, type=int,
                               help="Number of synthetic train sample. Default: 1000.")
        _ds_group.add_argument('--synthetic-test-samples', default=100, type=int,
                               help="Number of synthetic train sample. Default: 100.")
        _ds_group.add_argument('--synthetic-input-shape', default="3,32,32", type=str,
                               help="Number of synthetic input shape (coma separated). Default: 3,32,32.")
        _ds_group.add_argument('--synthetic-output-shape', default="10", type=str,
                               help="Number of synthetic input shape (coma separated). Default: 10.")
        _ds_group.add_argument('--test-as-validation', default=False, type=bool_lambda,
                               help="Prevent making partitions on training data for training+validation data, use test data for validation. True if specified.")
        _ds_group.add_argument('--validation-split', type=factor, default=0.2,
                               help="Split between training and validation data.")
        _ds_group.add_argument('--augment-shuffle', default=True, type=bool_lambda,
                               help="Shuffle training images. Default: True.")
        _ds_group.add_argument('--augment-flip', default=False, type=bool_lambda,
                               help="Flip horizontally training images. Default: False.")
        _ds_group.add_argument('--augment-flip-prob', type=factor, default=0.5,
                               help="Probability to flip training images. Default: 0.5.")
        _ds_group.add_argument('--augment-crop', default=False, type=bool_lambda,
                               help="Crop training images. Default: False.")
        _ds_group.add_argument('--augment-crop-size', type=int, default=16,
                               help="Size to crop training images. Default: 16.")
        _ds_group.add_argument('--augment-crop-prob', type=factor, default=0.5,
                               help="Probability to crop training images. Default: 0.5.")
        _ds_group.add_argument('--transform-crop', default=False, type=bool_lambda,
                               help="Crop the images. True if specified.")
        _ds_group.add_argument('--transform-crop-perc', type=float, default=0.875,
                               help="Central crop of the images. Default: 0.875.")
        _ds_group.add_argument('--transform-resize', default=False, type=bool_lambda,
                               help="Resize the images. True if specified.")
        _ds_group.add_argument('--transform-resize-size', type=int, default=300,
                               help="New size of the images. Default: 300.")
        _ds_group.add_argument('--normalize', default=False, type=bool_lambda,
                               help="Normalize dataset. Default: False.")
        _ds_group.add_argument('--normalize-offset', type=float, default=-0.45,
                               help="Offset samples by a value. Default: -0.45.")
        _ds_group.add_argument('--normalize-scale', type=float, default=3.75,
                               help="Scale samples by a value. Default: 3.75.")

        # Optimization options
        _oo_group = self.add_argument_group("Optimization options")
        _oo_group.add_argument('--enable-best-of', type=bool_lambda, default=False,
                               help="Enable the BestOf auto-tuner.")
        _oo_group.add_argument('--enable-memory-cache', type=bool_lambda, default=True,
                               help="Enable the memory cache module to use persistent memory.")
        _oo_group.add_argument('--enable-fused-bn-relu', type=bool_lambda, default=False,
                               help="Fuse BatchNormalization and Relu layers. True if specified.")
        _oo_group.add_argument('--enable-fused-conv-relu', type=bool_lambda, default=False,
                               help="Fuse Conv2D and Relu layers. True if specified.")
        _oo_group.add_argument('--enable-fused-conv-bn', type=bool_lambda, default=False,
                               help="Fuse Conv2D and BatchNormalization layers. True if specified.")
        _oo_group.add_argument('--enable-fused-conv-bn-relu', type=bool_lambda, default=False,
                               help="Fuse Conv2D and BatchNormalization and Relu layers. Default: False.")

        # Convolution methods
        _cm_group = self.add_argument_group("Convolution options")
        _cm_group.add_argument('--conv-variant', type=str, default="i2c", choices=["i2c", "gemm", "winograd", "direct"],
                               help="Select the standard 2D Convolutional module. Options: \n" \
                                    "* \'i2c\': Use the ConvI2C algorithm. \n" \
                                    "* \'gemm\': Use the ConvGemm algorithm. \n" \
                                    "* \'winograd\': Use the CondWinograd algorithm. \n" \
                                    "* \'direct\': Use the ConvDirect algorithm. \n" \
                                    "Default: \"i2x\".")
        _cm_group.add_argument('--conv-direct-method', type=str, default="",
                               help="Use ConvDirect module to realize convolutions in Conv2D layers. True if specified.")
        _cm_group.add_argument('--conv-direct-methods-for-best-of', type=str, default="",
                               help="ConvDirect modules to compare in \'best_of\' option if specified.")

        # Optimizer options
        _op_group = self.add_argument_group("Optimizer options")
        _op_group.add_argument('--optimizer', dest="optimizer_name", type=str, default="sgd", choices=["sgd", "rmsprop", "adam", "nadam", "oktopk"],
                               help="Optimizers: \'sgd\', \'rmsprop\', \'adam\', \'nadam\', \'oktopk\'. Default: \'sgd\'. ")
        _op_group.add_argument('--learning-rate', type=float, default=1e-2,
                               help="Learning rate. Default: 0.01.")
        _op_group.add_argument('--learning-rate-scaling', default=False, type=bool_lambda,
                               help="Scale learning rate in data parallelism: new_lr = lr / num_procs.  True if specified.")
        _op_group.add_argument('--optimizer-momentum', type=float, default=0.9,
                               help="Decay rate for \'sgd\' optimizer. Default: 0.9. optimizers. Default: 1e-8.")
        _op_group.add_argument('--optimizer-decay', type=float, default=0.0,
                               help="Decay rate for optimizers. Default: 0.0.")
        _op_group.add_argument('--optimizer-nesterov', default=False, type=bool_lambda,
                               help="Whether to apply Nesterov momentum. Default: False.")
        _op_group.add_argument('--optimizer-beta1', type=float, default=0.99,
                               help="Variable for \'adam\', \'nadam\' optimizers. Default: 0.99.")
        _op_group.add_argument('--optimizer-beta2', type=float, default=0.999,
                               help="Variable for \'adam\', \'nadam\' optimizers. Default: 0.999.")
        _op_group.add_argument('--optimizer-epsilon', type=float, default=1e-7,
                               help="Variable for \'rmsprop\', \'adam\', \'nadam\'. Default=1e-7.")
        _op_group.add_argument('--optimizer-rho', type=float, default=0.9,
                               help="Variable for \'rmsprop\' optimizers. Default: 0.99.")
        _op_group.add_argument('--optimizer-tau', type=int, default=64,
                               help="Variable for \'oktopk\' optimizers. Default: 64.")
        _op_group.add_argument('--optimizer-tau-prime', type=int, default=32,
                               help="Variable for \'oktopk\' optimizers. Default: 32.")
        _op_group.add_argument('--optimizer-density', type=float, default=0.01,
                               help="Variable for \'oktopk\' optimizers. Default: 0.01.")
        _op_group.add_argument('--loss-func', dest="loss_func_name", type=str, default="categorical_cross_entropy",
                               choices=["categorical_cross_entropy", "binary_cross_entropy"],
                               help="Loss functions that is evaluated on each trained batch: \'categorical_cross_entropy\', \'binary_cross_entropy\'. Default \'categorical_cross_entropy\'.")
        _op_group.add_argument(
            '--metrics',
            type=str,
            default="categorical_accuracy",
            help="List of comma-separated metrics that are evaluated on each trained batch: \'categorical_accuracy\', \'categorical_hinge\', \'categorical_mse\', \'categorical_mae\', \'regression_mse\', \'regression_mae\', \'binary_confusion_matrix\', \'multiclass_confusion_matrix\', \'precision\', \'recall\', \'f1_score\'. Default: \'categorical_accuracy\'.")

        # Schedulers options
        _sh_group = self.add_argument_group("Schedulers options")
        _sh_group.add_argument('--schedulers', dest="schedulers_names", type=str,
                               default="early_stopping,reduce_lr_on_plateau,model_checkpoint",
                               help="List of comma-separated LR schedulers: \'warm_up\', \'early_stopping\', \'reduce_lr_on_plateau\', \'reduce_lr_every_nepochs\', \'model_checkpoint\'. Default: \'early_stopping,reduce_lr_on_plateau,model_checkpoint\'.")
        _sh_group.add_argument('--warm-up-epochs', type=int, default=5,
                               help="Number of batches (ramp up) that the LR is scaled up from 0 until LR. Default: 5.")
        _sh_group.add_argument('--early-stopping-metric', type=str, default="val_categorical_cross_entropy",
                               help="Loss metric monitored by early_stopping LR scheduler. Default: \'val_categorical_cross_entropy\'.")
        _sh_group.add_argument('--early-stopping-patience', type=int, default=10,
                               help="Number of epochs with no improvement after which training will be stopped. Default: 10.")
        _sh_group.add_argument('--early-stopping-minimize', type=bool_lambda, default=True,
                               help="Whether to minize the metric. If False, it will maximize. Default: True.")
        _sh_group.add_argument('--reduce-lr-on-plateau-metric', type=str, default="val_categorical_cross_entropy",
                               help="Loss metric monitored by reduce_lr_on_plateau LR scheduler. Default: \'val_categorical_cross_entropy\'.")
        _sh_group.add_argument('--reduce-lr-on-plateau-factor', type=float, default=0.1,
                               help="Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1.")
        _sh_group.add_argument('--reduce-lr-on-plateau-patience', type=int, default=5,
                               help="Number of epochs with no improvement after which LR will be reduced. Default: 5.")
        _sh_group.add_argument('--reduce-lr-on-plateau-min-lr', type=float, default=0,
                               help="Lower bound on the learning rate. Default: 0.")
        _sh_group.add_argument('--reduce-lr-every-nepochs-factor', type=float, default=0.1,
                               help="Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1.")
        _sh_group.add_argument('--reduce-lr-every-nepochs-nepochs', type=int, default=5,
                               help="Number of epochs after which LR will be periodically reduced. Default: 5.")
        _sh_group.add_argument('--reduce-lr-every-nepochs-min-lr', type=float, default=0,
                               help="Lower bound on the learning rate. Default: 0.")
        _sh_group.add_argument('--stop-at-loss-metric', type=str, default="val_accuracy",
                               help="Loss metric monitored by stop_at_loss LR scheduler. Default: \'val_accuracy\'.")
        _sh_group.add_argument('--stop-at-loss-threshold', type=float, default=0,
                               help="Metric threshold monitored by stop_at_loss LR scheduler. Default: 0.")
        _sh_group.add_argument('--model-checkpoint-metric', type=str, default="val_categorical_cross_entropy",
                               help="Loss metric monitored by model_checkpoint LR scheduler. Default: \'val_categorical_cross_entropy\'")
        _sh_group.add_argument('--model-checkpoint-save-freq', type=int, default=2,
                               help="Frequency (in epochs) at which the model weights and bias will be saved by the model_checkpoint LR scheduler. Default: 2.")

        # Parallel execution options
        _pe_group = self.add_argument_group("Parallel execution options")
        _pe_group.add_argument('--parallel', type=str, default="sequential", choices=["sequential", "data"],
                               help="Data parallelization modes: \'sequential\', \'data\' (MPI). Default: \'sequential\'.")
        _pe_group.add_argument('--use-blocking-mpi', type=bool_lambda, default=True,
                               help="Enable non-blocking MPI primitives. Default: True.")
        _pe_group.add_argument('--use-mpi-buffers', type=bool_lambda, default=None,
                               help="Enable the use of MPI buffers. Possible values: \'True\' (MPI operations by buffer), \'False\' (MPI operations by object) or undefined (auto-select the better option). Default: undefined.")
        _pe_group.add_argument('--enable-gpu', type=bool_lambda, default=False,
                               help="Enable GPU, use cuDNN library. Default: False.")
        _pe_group.add_argument('--enable-gpudirect', type=bool_lambda, default=False,
                               help="Enable GPU pinned memory for gradients when using a CUDA-aware MPI version. Default: False.")
        _pe_group.add_argument('--enable-nccl', type=bool_lambda, default=False,
                               help="Enable the use of the NCCL library for  collective communications on GPUs. This option can only be set  with \'--enable-gpu\'. Default. False.")
        _pe_group.add_argument('--enable-cudnn-auto-conv-alg', type=bool_lambda, default=True,
                               help="Let cuDNN to select the best performing convolution algorithm. Default: True.")

        # Encryption options
        _cy_group = self.add_argument_group("Encryption options")
        _cy_group.add_argument('--encryption', dest="encryption_name", type=str, default="",
                               help="Encryption library: \'tenseal\', \'openfhe\', \'\' (None). Default \'\' (None).")
        _cy_group.add_argument('--encryption-poly-degree', type=int, default=13,
                               help="Encryption polynomial degree. 2 ^ \'value\'. Default: 13.")
        _cy_group.add_argument('--encryption-global-scale', type=int, default=40,
                               help="Encryption global scale. 2 ^ \'value'\'. Default: 40.")
        _cy_group.add_argument('--encryption-security-level', type=int, default=128,
                               help="Encryption security level: 0 (Not set), 128, 192, 256. Default: 128.")

        # Tracing and profiling
        _tr_group = self.add_argument_group("Tracing options")
        _tr_group.add_argument('--tracing', type=bool_lambda, default=False,
                               help="Obtain Simple/Extrae-based traces. Deffault: False.")
        _tr_group.add_argument('--tracer-output', type=str, default="",
                               help="Output file to store the Simple/Extrae-based traces.")
        _tr_group.add_argument('--tracer-pmlib-server', type=str, default="127.0.0.1",
                               help="Address of PMlib tracer server. Default: \'127.0.0.1\'.")
        _tr_group.add_argument('--tracer-pmlib-port', type=int, default=6526,
                               help="Port of PMlib tracer server. Default: 6526.")
        _tr_group.add_argument('--tracer-pmlib-device', type=str, default="",
                               help="Port of PMlib tracer device.")
        _tr_group.add_argument('--profile', type=bool_lambda, default=False,
                               help="Obtain cProfile profiles. Default: False.")

        # Performance modeling options
        _pm_group = self.add_argument_group("Performance modeling options")
        _pm_group.add_argument('--cpu-speed', type=float, default=4e12, help=argparse.SUPPRESS)
        _pm_group.add_argument('--memory-bw', type=float, default=50e9, help=argparse.SUPPRESS)
        _pm_group.add_argument('--network-bw', type=float, default=1e9, help=argparse.SUPPRESS)
        _pm_group.add_argument('--network-lat', type=float, default=0.5e-6, help=argparse.SUPPRESS)
        _pm_group.add_argument('--network-alg', type=str, default="vdg", choices=["bta", "vdg"], help=argparse.SUPPRESS)

        # Add Runtime parallel execution options
        _re_group = self.add_argument_group("Runtime parallel execution options")
        _re_group.add_argument('--mpi-processes', type=int, default=-1, help=argparse.SUPPRESS)
        _re_group.add_argument('--threads-per-process', type=int, default=-1, help=argparse.SUPPRESS)
        _re_group.add_argument('--gpus-per-node', type=int, default=-1, help=argparse.SUPPRESS)

        # Add Communication options
        _cm_group = self.add_argument_group("Communication options")
        _cm_group.add_argument('--mpi-protocol', type=str, default="", help=argparse.SUPPRESS)
        _cm_group.add_argument('--mpi-server', type=str, default="", help=argparse.SUPPRESS)
        _cm_group.add_argument('--mpi-port', type=int, default=-1, help=argparse.SUPPRESS)

    def parse_args(self, args=None, namespace=None):
        # Call super.parse_args
        result = super().parse_args(args, namespace)
        # Add runtime data
        result.mpi_processes = _get_mpi_processes()
        result.threads_per_process = _get_threads_per_process()
        result.gpus_per_node = _get_gpus_per_node()
        result.mpi_protocol = _get_mpi_protocol()
        result.mpi_server = _get_mpi_server()
        result.mpi_port = _get_mpi_port()
        # Populate self.lines (for self.print_args())
        if len(self.lines) == 0:
            lines = []
            for action_group in self._action_groups:
                indent = ""
                length = 0
                if action_group.title not in ('positional arguments', 'optional arguments'):
                    indent = "  "
                    lines.append("")
                    lines.append(action_group.title)
                    if action_group.description is not None:
                        lines.append(action_group.description)
                for action in action_group._group_actions:
                    if action.default == '==SUPPRESS==':
                        continue
                    option_string = f"{action.option_strings[0].replace('--', '')}"
                    if len(option_string) > length:
                        length = len(option_string)
                for action in action_group._group_actions:
                    if action.default == '==SUPPRESS==':
                        continue
                    option_string = f"{action.option_strings[0].replace('--', '')}"
                    tab = " " * (length - len(option_string))
                    lines.append(f"{indent}{option_string}{tab}: {getattr(result, action.dest)}")
            lines.append('')
            self.lines = lines
        return result

    def print_args(self) -> None:
        print("\n".join(self.lines))

    def get_default_values(self) -> dict[str, Any]:
        return vars(self.parse_args([]))

    def to_dict(self) -> dict[str, Any]:
        return vars(self.parse_args())
