"""
PyDTNN model
"""

#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021-25 Universitat Jaume I
#
#  PyDTNN is free software: you can redistribute it and/or modify it under the
#  terms of the GNU General Public License as published by the Free Software
#  Foundation, either version 3 of the License, or (at your option) any later
#  version.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT
#  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
#  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
#  License for more details.
#
#  You should have received a copy of the GNU General Public License along
#  with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import functools
import importlib
import os
import sys
import time
from timeit import default_timer as timer

# Typing-related import
from typing import Any, TypeVar
from collections.abc import Iterable
from .tracers import SimpleTracerGPU

from types import ModuleType
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase
from .tracers.tracer import Tracer
from .datasets import Dataset

from tqdm import tqdm
import numpy as np

import pydtnn.metrics
from pydtnn.utils import PYDTNN_TENSOR_FORMAT_NHWC, PYDTNN_TENSOR_FORMAT_NCHW
from . import losses, metrics
from . import utils
from .datasets import CustomDataset, get_dataset
from .lr_schedulers import get_lr_schedulers
from .optimizers import get_optimizer
from .parser import parser
from .performance_models import *
from .tracers import PYDTNN_MDL_EVENT, PYDTNN_MDL_EVENTS, PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, \
    PYDTNN_EVENT_FINISHED, ExtraeTracer, SimpleTracer, PYDTNN_MDL_EVENT_enum, PYDTNN_OPS_EVENT_enum    
from .utils.best_of import BestOf
from .utils.memory_cache import MemoryCache
from .utils.performance_counter import PerformanceCounter


# --- CUDA related imports --- #
import atexit
cuda_error_msg = list()
try:
    import pydtnn.backends.gpu.tensor_gpu
    # noinspection PyUnresolvedReferences
    import pycuda.gpuarray as gpuarray
except (ImportError, ModuleNotFoundError) as e: 
    gpuarray = None
    cuda_error_msg.append(e)
except Exception as e: raise(e)
try:
    # noinspection PyUnresolvedReferences
    import pycuda.driver as drv
    from pydtnn.backends.gpu.libs import libcudnn as cudnn
except (ImportError, ModuleNotFoundError) as e: 
    drv = None
    cuda_error_msg.append(e)
except Exception as e: raise(e)
try: 
    # noinspection PyUnresolvedReferences
    from skcuda import cublas
except (ImportError, ModuleNotFoundError) as e: 
    cublas = None
    cuda_error_msg.append(e)
except Exception as e: raise(e)
# --- END CUDA related imports --- #

# --- GLOBAL VARIABLES --- #
supported_gpu: bool = False
supported_cudnn: bool = True
supported_nccl: bool = True
enable_cudnn: bool = False
# --- END GLOBAL VARIABLES --- #

try:
    # noinspection PyUnresolvedReferences,PyPackageRequirements
    from pydtnn.libs.mpi import MPI
except (ImportError, ModuleNotFoundError):
    MPI = None

# --- CONSTANS --- #
BAR_WIDTH = 140
EVALUATE_MODE, TRAIN_MODE, UNSPECIFIED_MODE = (0, 1, 2)
DEFAULT_BACH_SIZE = 64
# --- END CONSTANS --- #

# TODO: Change ModuleType to the actual module type.
NCCL_DataType = TypeVar("NCCL_DataType") # NOTE: Check "_initialize_cuda" to get the actual types.
NCCL_Comm_Type = TypeVar("NCCL_Comm_Type") # NOTE: Check "_initialize_cuda" to get the actual types.
Cudnn_Handle_Type = TypeVar("Cudnn_Handle_Type") # NOTE: Check "_initialize_cuda" to get the actual types.
Cublas_Handle_Type = TypeVar("Cublas_Handle_Type") # NOTE: Check "_initialize_cuda" to get the actual types.
PyCuda_Stream_Type  = TypeVar("PyCuda_Stream_Type") # NOTE: Check "_initialize_cuda" to get the actual types.
Cudnn_dtype = TypeVar("Cudnn_dtype") # NOTE: Check "_initialize_cuda" to get the actual types.

def _layer_id_generator() -> Iterable[int]:
    """To obtain consecutive layer ids. See Layer.set_model()."""
    current_layer_id = 0
    while True:
        yield current_layer_id
        current_layer_id += 1
# --- END _layer_id_generator --- #

# TODO: Check the output.
def ensure_model_is_initialized(method):
    @functools.wraps(method)
    def wrapper_ensure_model_is_initialized(*args, **kwargs):
        self = args[0]
        if not self._initialized:
            self._initialize()
        return method(*args, **kwargs)

    return wrapper_ensure_model_is_initialized
# --- END ensure_model_is_initialized --- #

def _initilize_communications(parallel: str) -> tuple[ModuleType | None, ModuleType | None]:
    
    
    match parallel:
        case "sequential":
            mpi = None
            comm = None
        case "data":
            if not MPI:
                raise SystemExit("Please, install mpi4py to allow parallel MPI execution!")
            mpi: ModuleType = MPI
            comm: ModuleType = MPI.COMM_WORLD
        case _:
            raise SystemExit(f"Parallel option '{parallel}' not recognized.")
    
    return (mpi, comm)
# --- END _initilize_communications --- #

def _set_execution_attributes(comm: ModuleType | None, shared_storage: bool) -> tuple[float, int, int, int, int, int]:

    rank_weight = 1.0
    comm_rank = rank = 0
    comm_size = nprocs = 1        
    if comm:
        # NOTE: "if MPI" was already check in "_initilize_communications"
        # FIXME: remove this if-else.
        if MPI:
            comm_rank:int = comm.Get_rank() # NOTE: From libs.mpi.client.py
            comm_size:int = comm.Get_size() # NOTE: From libs.mpi.client.py
            if shared_storage:
                rank = comm_rank
                nprocs = comm_size
            #else: Nothing each rank is independant
        else:
            raise SystemExit("Please, install mpi4py to allow parallel MPI execution!")
    comm_groups = comm_size // nprocs

    return rank_weight, comm_rank, comm_size, rank, nprocs, comm_groups
# --- END _set_execution_attributes --- #

def _initilize_and_get_tracer(tracer_output: str, tracing: bool, comm: ModuleType, enable_gpu: bool, 
                              tracer_pmlib_server:str, tracer_pmlib_port:int, tracer_pmlib_device:str) -> Tracer:
    
    if tracer_output == "":
        tracer = ExtraeTracer(tracing)
    else:
        if enable_gpu:
            tracer = SimpleTracerGPU(tracing, tracer_output, comm)
        else:
            if tracer_pmlib_device != "":
                from .tracers import SimpleTracerPMLib
                tracer = SimpleTracerPMLib(tracing, tracer_output, comm,
                                                tracer_pmlib_server, tracer_pmlib_port, tracer_pmlib_device)
            else:
                tracer = SimpleTracer(tracing, tracer_output, comm)
    return tracer
# --- END _initilize_and_get_tracer --- #

def _initialize_cuda(comm: ModuleType, comm_rank: int, rank:int, nprocs:int, 
                     gpus_per_node:int, parallel:str, dtype: np.dtype, 
                     enable_nccl: bool, tracer: SimpleTracer) -> tuple[NCCL_DataType | None, NCCL_Comm_Type | None, 
                                                                 Cudnn_Handle_Type, Cublas_Handle_Type, 
                                                                 PyCuda_Stream_Type, Cudnn_dtype]:
        
    global supported_cudnn, supported_nccl
    supported_cudnn = True
    supported_nccl = True
    
    # TODO | FIXME: Remove the following comments after the commit.
    # import pycuda.autoinit
    # The next fake test exists only to avoid the pycuda.autoinit import being removed when optimizing imports
    # if self.kwargs.get('fake_pycuda_autoinit_option'):
    #    pycuda.autoinit()
    # Uncomment the next code if pycuda.autoinit is not available

    device_id:int = comm_rank % drv.Device.count()
    drv.init()
    context:int = drv.Device(device_id).make_context()
    
    atexit.register(context.pop)
    
    nccl_type = None
    nccl_comm = None

    if comm and enable_nccl:
        try:
            from pydtnn.backends.gpu.libs import libnccl as nccl
        except (ImportError, ModuleNotFoundError, OSError):
            supported_nccl = False
            msg = "Please, install nccl to be able to use NVIDIA NCCL inter-GPU communications!"
            raise SystemExit(msg) from None

        nccl_types = {np.float64: nccl.DataType.Float64, 
                      np.float32: nccl.DataType.Float32, 
                      np.int8: nccl.DataType.Int8, 
                      np.int32: nccl.DataType.Int32}

        nccl_type: NCCL_DataType = nccl_types.get(dtype, nccl.DataType.Float32)

        hostname = MPI.Get_processor_name()

        hosts_data = comm.allgather([rank, hostname])
        # Build a dictionary hostname : [ranks_in_host]
        #   { "host1": [0, 1], "host2": [2, 3] }
        hosts = {}
        for r, h in hosts_data:
            # noinspection PyTypeChecker
            hosts.setdefault(h, []).append(r)
        if parallel == "data":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % gpus_per_node)
        # Check that no more processes than GPUs per node are used
        for host, ranks_in_host in hosts.items():
            if len(ranks_in_host) > gpus_per_node:
                raise SystemExit("Not able to run more processes than GPUs per node!")

        nccl_id = comm.bcast(nccl.ncclGetUniqueId() if comm_rank == 0 else None)
        nccl_comm: NCCL_Comm_Type = nccl.ncclCommInitRank(nprocs, nccl_id, rank)

    cudnn_handle: Cudnn_Handle_Type = cudnn.cudnnCreate()
    cublas_handle: Cublas_Handle_Type = cublas.cublasCreate()
    stream: PyCuda_Stream_Type = drv.Stream()
    cublas.cublasSetStream(cublas_handle, stream.handle)
    cudnn.cudnnSetStream(cudnn_handle, stream.handle)

    cudnn_types = {np.float64: "CUDNN_DATA_DOUBLE", 
                   np.float32: "CUDNN_DATA_FLOAT", 
                   np.int8: "CUDNN_DATA_INT8", 
                   np.int32: "CUDNN_DATA_INT32"}

    cudnn_type:str = cudnn_types.get(dtype, "CUDNN_DATA_FLOAT")

    cudnn_dtype: Cudnn_dtype = cudnn.cudnnDataType[cudnn_type]
    tracer.set_default_stream(stream)
        
    return nccl_type, nccl_comm, cudnn_handle, cudnn_handle, cublas_handle, stream, cudnn_dtype
# --- END _initialize_cuda --- #

def _set_data_format(tensor_format:str, enable_cudnn:bool) -> int:
    match tensor_format:
        case "AUTO":
            tensor_format = PYDTNN_TENSOR_FORMAT_NCHW if enable_cudnn else PYDTNN_TENSOR_FORMAT_NHWC
        case "NCHW":
            tensor_format = PYDTNN_TENSOR_FORMAT_NCHW
        case _: # case "NHWC":
            tensor_format = PYDTNN_TENSOR_FORMAT_NHWC
    return tensor_format
# --- END _set_data_format --- #

def _calculate_batch_size(batch_size: int | None, global_batch_size: int | None, comm_size: int) -> int:

    if batch_size and global_batch_size:
        raise SystemExit("Can not define 'batch_size' and 'global_batch_size' simultaneously")
    
    if global_batch_size:
        # NOTE: Using comm_size instead of nprocs might not be appropriate, 
        #       as it differs to how global_batch_size is defined elsewhere,
        #       but for now it just a parser option difference that helps testing
        _batch_size = global_batch_size // comm_size
    elif batch_size:

        _batch_size = batch_size
    else:
        _batch_size = DEFAULT_BACH_SIZE

    # NOTE | TODO: Check if '(num processes: {comm_size})'
    if _batch_size < 1:
        raise SystemExit(f"'batch_size' ({_batch_size}) too small or too many processes (num processes: {comm_size})")

    return _batch_size
# --- END _calculate_batch_size --- #

class Model:
    """
    PyDTNN Model
    """

    def __init__(self, parallel:str="sequential", non_blocking_mpi:bool=False, enable_gpu:bool=False, enable_gpudirect:bool=False,
                 enable_nccl:bool=False, dtype:np.dtype=np.float32, tracing: bool=False, tracer_output:str="",
                 tracer_pmlib_server:str="127.0.0.1", tracer_pmlib_port:int=6526, tracer_pmlib_device:str="",
                 **kwargs):
        # Attributes related to the given arguments
        self.parallel: bool = parallel
        self.blocking_mpi: bool = not non_blocking_mpi
        global enable_cudnn
        enable_cudnn = self.enable_cudnn = enable_gpu
        self.gpudirect:bool = enable_gpudirect
        self.enable_nccl:bool = enable_nccl
        self.dtype:np.dtype = dtype
        
        self.nparams:int = 0 # NOTE: Model's total number of params
        
        # The following attributes will be initilized later only if "enable_cudnn" is True.
        self.nccl_type: NCCL_DataType | None = None
        self.nccl_comm: NCCL_Comm_Type | None = None
        self.cudnn_handle: Cudnn_Handle_Type | None = None
        self.cublas_handle: Cublas_Handle_Type | None = None
        self.stream: PyCuda_Stream_Type | None = None
        self.cudnn_dtype: Cudnn_dtype | None = None

        # FIXME | TODO: Get the parser's value from other place.
        # Get default values from parser and update them from the received kwargs
        self.kwargs: dict[str, Any] = vars(parser.parse_args([]))
        self.kwargs.update(kwargs)
        
        self.shared_storage:bool = self.kwargs["shared_storage"] # NOTE: This parameter comes from "Parser" (vars(parser.parse_args([])))
        
        # Set MPI and comm          
        self.MPI, self.comm = _initilize_communications(parallel = parallel)
        
        # Execution attributes
        (self.rank_weight, self.comm_rank, self.comm_size, 
         self.rank, self.nprocs, self.comm_groups) = _set_execution_attributes(comm = self.comm, shared_storage = self.shared_storage)        
        
        # Set tracer
        self.tracer = _initilize_and_get_tracer(tracer_output = tracer_output, tracing = tracing, comm = self.comm, enable_gpu = enable_gpu,
                                                tracer_pmlib_server = tracer_pmlib_server, tracer_pmlib_port = tracer_pmlib_port, 
                                                tracer_pmlib_device = tracer_pmlib_device)

        # Set performance counter
        self.perf_counter = PerformanceCounter()
        
        # Layers' attributes
        self.layers: list[LayerAndActivationBase] = []
        self.layer_id:int = _layer_id_generator()
        self.shared_storage:bool = self.kwargs["shared_storage"] # NOTE: This parameter comes from "Parser" (vars(parser.parse_args([])))
        
        # Matmul
        self.matmul = utils.matmul
        
        # Set current mode to unspecified
        self.mode:int = UNSPECIFIED_MODE

        # Memory cache optimization
        self.enable_memory_cache: bool # NOTE: This parameter comes from "Parser" (vars(parser.parse_args([])))
        if self.enable_memory_cache:
            MemoryCache.enable()
        else:
            MemoryCache.disable()        
        
        # Cuda
        if self.enable_cudnn:
            if gpuarray and drv and cublas:
                self.gpus_per_node: int # NOTE: This parameter comes from "Parser" (vars(parser.parse_args([])))
                (self.nccl_type, self.nccl_comm, self.cudnn_handle, 
                 self.cublas_handle, self.stream, self.cudnn_dtype) = _initialize_cuda(comm = self.comm, comm_rank = self.comm_rank, rank = self.rank,
                                                                              nprocs = self.nprocs, gpus_per_node = self.gpus_per_node, 
                                                                              parallel = self.parallel, dtype = self.dtype, enable_nccl = 
                                                                              self.enable_nccl, tracer = self.tracer)
            else:
                raise ImportError(" ".join(cuda_error_msg))
        else: cuda_error_msg = None # If CUDA is not going to be used, then the import errors should be deleted (or mark to be deleted).
        
        # Data format
        self.tensor_format = _set_data_format(tensor_format = self.tensor_format, enable_cudnn = self.enable_cudnn)

        # Disable BestOf globally if not enabled
        if self.kwargs['enable_best_of'] is False: 
            # NOTE: comes from "Parser" (vars(parser.parse_args([])))
            BestOf.use_always_the_first_alternative()
        
        self.batch_size = _calculate_batch_size(batch_size = self.kwargs['batch_size'], # NOTE: This parameters comes from "Parser" (vars(parser.parse_args([])))
                                                global_batch_size = self.kwargs['global_batch_size'], # NOTE: This parameters comes from "Parser" (vars(parser.parse_args([])))
                                                comm_size= self.comm_size)
        
        # Explicit declaration of those model attributes that are referenced by other parts of PyDTNN
        #   NOTE: The following parameters come from "Parser" (vars(parser.parse_args([])))
        self.steps_per_epoch = self.kwargs['steps_per_epoch']
        self.cpu_speed = self.kwargs['cpu_speed']
        self.memory_bw = self.kwargs['memory_bw']
        self.network_bw = self.kwargs['network_bw']
        self.network_lat = self.kwargs['network_lat']
        self.network_alg = self.kwargs['network_alg']

        self.weights_and_bias_filename:str # NOTE: This parameter comes from "Parser" (vars(parser.parse_args([])))
        # Load weights and bias
        if self.weights_and_bias_filename:
            self.load_weights_and_bias(self.weights_and_bias_filename)
        # Dataset
        self.dataset: Dataset = get_dataset(self)
        
        # ¡¡¡TODO: POR AQUÍ!!!
        # NOTE: Check where 'self.kwargs["learning_rate"]' is used.

        # Optimizers and LRSchedulers
        # NOTE: 'self.kwargs["learning_rate_scaling"]' comes from "Parser" (vars(parser.parse_args([])))
        if self.kwargs["learning_rate_scaling"]:
            # using comm_size instead of nprocs might not be appropriate,
            # as it differs to how learning_rate is defined elsewhere,
            # but for now it just a parser option difference that helps testing
            self.kwargs["learning_rate"] /= self.comm_size
        self.optimizer = get_optimizer(self)
        self.lr_schedulers = get_lr_schedulers(self)
        # Metrics list
        self.metrics_list = [m for m in self.metrics.replace(" ", "").split(",")]
        # Private attributes
        self._evaluate_round = 0
        self._initialized = False
        # Attributes that will be properly defined elsewhere
        self.y_batch = None
        self.history = None
        # Read the model (must be the last action, as it calls self._initialize() if there is a model)
        self.model_name = self.kwargs.get("model_name")
        if self.model_name:
            self._read_model(self.model_name)
        # Syncronization parameters
        self.comm_nsamples = self.comm.allgather(self.dataset.train_nsamples) if self.comm else [self.dataset.train_nsamples]
        if self.model_sync_alg not in {"avg", "wavg", "invwavg"}:
            raise SystemExit(f"Process weight option '{self.proc_weight}' not recognized.")
        if self.model_sync_participation not in {"all", "avail2all"}:
            raise SystemExit(f"Process weight option '{self.proc_weight}' not recognized.")

    @property
    def dataset_raw_path(self):
        """Raw dataset path with rank substituted"""
        return utils.string_substitute(self.kwargs["dataset_raw_path"], rank=self.comm_rank)

    def __getattr__(self, item):
        try:
            return self.kwargs[item]
        except KeyError:
            raise AttributeError(f"Model object has no attribute '{item}'!") from None

    def _read_model(self, model_name):
        try:
            model_module = importlib.import_module(f"pydtnn.models.{model_name}")
            getattr(model_module, f"create_{model_name}")(self)
        except (ModuleNotFoundError, AttributeError):
            import traceback
            print(traceback.format_exc())
            sys.exit(-1)
        else:  # There was no error, call _initialize()
            self._initialize()

    def show(self):
        bfp = {np.float32: 4, np.float64: 8}[self.dtype]
        line = "+-------+--------------------------+---------+---------------+-------------------" \
               "+-------------------------------------+"
        head = "| Layer |           Type           | #Params | Output shape  |   Weights shape   " \
               "|             Parameters              |"
        print(line)
        print(head)
        for layer in self.layers:
            print(line)
            layer.show()
        print(line)
        print(f"|{'':^7s} {'Total parameters':^26s} {self.nparams:^9d} {utils.convert_size_bytes(self.nparams * bfp):^15s} "
              f"{'':19s} {'':37s}|")
        print(line)

    def print_in_convdirect_format(self):
        line = "#l\tkn\two\tho\tt\tkh\tkw\tci\twi\thi"
        print(line)
        for layer in self.layers:
            layer.print_in_convdirect_format()

    def add(self, layer):
        layer.set_model(self)
        need_dx = layer.id > 0
        prev_shape = self.layers[-1].shape if layer.id > 0 else ()
        if self.enable_cudnn:
            y = self.layers[-1].y if layer.id > 0 else None
            layer.initialize(prev_shape, need_dx, y)
        else:
            layer.initialize(prev_shape, need_dx)

        self.nparams += layer.nparams
        self.layers.append(layer)

        if layer.act:
            self.add(layer.act())

    def get_all_layers(self, from_layers=None):
        if from_layers is None:
            from_layers = self.layers
        this_recursion_layers = []
        for layer in from_layers:
            this_recursion_layers.append(layer)
            this_recursion_layers += self.get_all_layers(layer.children)
        return this_recursion_layers

    def _apply_layer_fusion(self, bn_relu=False, conv_relu=False, conv_bn=False, conv_bn_relu=False):
        """ Apply layer fusion in a recursive manner """

        def __layer_fusion(layers, bn_relu=False, conv_relu=False, conv_bn=False, conv_bn_relu=False):
            fused_layers = []
            for i, curr_layer in enumerate(layers):
                # if i > 0: print(i, curr_layer.canonical_name, fused_layers[-1].canonical_name)
                if curr_layer.is_block_layer:
                    for j, p in enumerate(curr_layer.paths):
                        curr_layer.paths[j] = __layer_fusion(p, bn_relu, conv_relu, conv_bn, conv_bn_relu)
                elif conv_bn_relu and len(fused_layers) > 1 and \
                        curr_layer.canonical_name == "Relu" and \
                        fused_layers[-1].canonical_name == "BatchNormalization" and \
                        fused_layers[-2].canonical_name == "Conv2D":
                    backend = "gpu" if self.enable_cudnn else "cpu"
                    fused_layer = getattr(importlib.import_module(f"pydtnn.backends.{backend}.layers"),
                                          fused_layers[-2].canonical_name +
                                          fused_layers[-1].canonical_name +
                                          type(curr_layer).__name__)
                    if fused_layers[-2].forward.__name__ in fused_layer.__dict__:  # or self.enable_best_of:
                        bn_layer = fused_layers.pop()
                        cv_layer = fused_layers.pop()
                        print("Fusing %03d_%s + %03d_%s + %03d_%s..." % (cv_layer.id, type(cv_layer).__name__,
                                                                         bn_layer.id, type(bn_layer).__name__,
                                                                         curr_layer.id, type(curr_layer).__name__))
                        curr_layer = fused_layer(from_parent=cv_layer, from_parent2=bn_layer)
                        curr_layer.initialize(from_parent_dict=cv_layer.__dict__)
                elif (conv_relu or conv_bn) and len(fused_layers) > 0 and \
                        (curr_layer.canonical_name == "Relu" or
                         curr_layer.canonical_name == "BatchNormalization") and \
                        fused_layers[-1].canonical_name == "Conv2D" and \
                        not (conv_bn_relu and i + 1 < len(layers) and layers[i + 1].canonical_name == "Relu"):
                    backend = "gpu" if self.enable_cudnn else "cpu"
                    fused_layer = getattr(importlib.import_module(f"pydtnn.backends.{backend}.layers"),
                                          fused_layers[-1].canonical_name +
                                          type(curr_layer).__name__)
                    if fused_layers[-1].forward.__name__ in fused_layer.__dict__:  # or self.enable_best_of:
                        prev_layer = fused_layers.pop()
                        print("Fusing %03d_%s + %03d_%s ..." % (prev_layer.id, type(prev_layer).__name__,
                                                                curr_layer.id, type(curr_layer).__name__))
                        curr_layer = fused_layer(from_parent=prev_layer, from_parent2=curr_layer)
                        curr_layer.initialize(from_parent_dict=prev_layer.__dict__)
                elif bn_relu and len(fused_layers) > 0 and \
                        curr_layer.canonical_name == "Relu" and \
                        fused_layers[-1].canonical_name == "BatchNormalization":
                    prev_layer = fused_layers.pop()
                    print("Fusing %03d_%s + %03d_%s ..." % (prev_layer.id, type(prev_layer).__name__,
                                                            curr_layer.id, type(curr_layer).__name__))
                    curr_layer = getattr(importlib.import_module("pydtnn.layers"),
                                         prev_layer.canonical_name +
                                         curr_layer.canonical_name)(from_parent=prev_layer)
                fused_layers.append(curr_layer)
            return fused_layers

        if not self.enable_cudnn and (bn_relu or conv_relu or conv_bn, conv_bn_relu):
            self.layers = __layer_fusion(self.layers, bn_relu, conv_relu, conv_bn, conv_bn_relu)

    def load_store_path(self, layers, d, mode):
        for layer in layers:
            name = layer.canonical_name
            if name in ["AdditionBlock", "ConcatenationBlock"]:
                for path in layer.paths:
                    self.load_store_path(path, d, mode)
            else:
                grad_vars = [g for g in layer.grad_vars] + \
                            (["running_var", "running_mean"] if name == "BatchNormalization" else [])
                if name == "BatchNormalization":
                    layer.updated_running_var = True
                for key in grad_vars:
                    base = f"{layer.id}_{name}_{key}"
                    if mode == "load" and base not in d:
                        print(f"Could not find '{base}' for layer '{name}' in file!")
                        continue
                    if mode == "load":
                        if self.enable_cudnn:
                            ary = getattr(layer, key).ary
                            ary.set(d[base].reshape(ary.shape))
                        else:
                            setattr(layer, key, d[base])
                    elif mode == "store":
                        if self.enable_cudnn:
                            d[base] = getattr(layer, key).ary.get()
                        else:
                            d[base] = getattr(layer, key)

    def load_weights_and_bias(self, filename):
        d = np.load(filename)
        self.load_store_path(self.layers, d, "load")

    def store_weights_and_bias(self, filename):
        if self.shared_storage and self.comm_rank == 0:
            d = {}
            self.load_store_path(self.layers, d, "store")
            np.savez_compressed(filename, **d)

    def calculate_time(self):
        # Total elapsed_time, Comp elapsed_time, Memo elapsed_time, Net elapsed_time
        total_time = np.zeros((4,), dtype=np.float32)

        # Forward pass (FP)
        for layer in range(1, len(self.layers)):
            total_time += self.layers[layer].fwd_time

        if self.blocking_mpi:
            # Blocking MPI
            # Back propagation. Gradient computation (GC) and weights update (WU)
            for layer in range(len(self.layers) - 1, 0, -1):
                total_time += self.layers[layer].bwd_time

            # Weight update (WU)
            for layer in range(len(self.layers) - 1, 0, -1):
                if self.comm and self.layers[layer].weights.size > 0:
                    total_time += allreduce_time(self.layers[layer].weights.size + self.layers[layer].biases.size,
                                                 self.cpu_speed, self.network_bw, self.network_lat,
                                                 self.network_alg, self.nprocs, self.dtype)
        else:
            total_time_iar = 0
            # Non-blocking MPI
            # Back propagation. Gradient computation (GC) and weights update (WU)
            for layer in range(len(self.layers) - 1, 0, -1):
                total_time += self.layers[layer].bwd_time
                if self.comm and self.layers[layer].weights.size > 0:
                    time_iar = allreduce_time(self.layers[layer].weights.size + self.layers[layer].biases.size,
                                              self.cpu_speed, self.network_bw, self.network_lat,
                                              self.network_alg, self.nprocs, self.dtype)
                    total_time[3] += time_iar[3]
                    total_time_iar = max(total_time[0], total_time_iar) + time_iar[0]

            total_time[0] = max(total_time[0], total_time_iar)

        return total_time

    def _compute_metrics_funcs(self, y_pred, y_targ, loss, blocking=True, comm=True):
        loss_req = None

        if y_targ.shape[0] > 0:
            if self.enable_cudnn:
                _losses = np.array([loss] + [func(y_pred.ary, y_targ.ary) for func in self.metrics_funcs], dtype=self.dtype)
            else:
                _losses = np.array([loss] + [func(y_pred, y_targ) for func in self.metrics_funcs], dtype=self.dtype)
        else:
            _losses = self.total_metrics.copy()
            _losses[0] = loss

        if self.comm is not None and comm:
            _losses /= self.comm_size
            if blocking:
                self.comm.Allreduce(MPI.IN_PLACE, _losses, op=MPI.SUM)
            else:
                loss_req = self.comm.Iallreduce(MPI.IN_PLACE, _losses, op=MPI.SUM)
        else:
            if blocking:
                pass
            else:
                raise NotImplementedError("can not compute metrics non-blocking locally")

        return _losses, loss_req

    def _update_running_average(self, curr, total, count, batch_size, prefix=""):
        string = ""
        total = ((curr * batch_size) + (total * count)) / (count + batch_size)
        for c in range(len(self.loss_and_metrics)):
            loss_str = pydtnn.metrics.metric_format.get(self.loss_and_metrics[c], self.loss_and_metrics[c])
            string += ("%s, " % (prefix + loss_str)) % total[c]
        string = string[:-2]
        return total, count + batch_size, string

    def _get_x_y_targ(self, x_batch, y_batch, current_batch_size):
        if self.enable_cudnn:
            if x_batch.shape[0] != current_batch_size:
                raise ValueError
            self.layers[0].y.ary.set(x_batch)
            self.y_batch.ary.set(y_batch)
            x, y_targ = self.layers[0].y, self.y_batch
        else:
            x, y_targ = x_batch, y_batch
        return x, y_targ

    def _zero_dx(self, y):
        return np.zeros_like(y, shape=(1,) + y.shape[1:])

    def _weight_update(self, gradient=True, blocking=True):
        if blocking:
            for i in range(len(self.layers) - 1, 0, -1):
                self.tracer.emit_event(PYDTNN_MDL_EVENT,
                                       self.layers[i].id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.ALLREDUCE_DW.value)
                self.layers[i].reduce_weights_sync(gradient=gradient)
                self.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)

        else:
            for i in range(len(self.layers) - 1, 0, -1):
                self.tracer.emit_event(PYDTNN_MDL_EVENT,
                                       self.layers[i].id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.ALLREDUCE_DW.value)
                self.layers[i].reduce_weights_async(gradient=gradient)
                self.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)

            for i in range(len(self.layers) - 1, 0, -1):
                self.tracer.emit_nevent([PYDTNN_MDL_EVENT, PYDTNN_OPS_EVENT],
                                        [self.layers[i].id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.WAIT_DW.value,
                                        self.layers[i].id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.OPS_ALLREDUCE_DW.value])
                self.layers[i].wait_allreduce_async(gradient=gradient)
                self.tracer.emit_nevent([PYDTNN_MDL_EVENT, PYDTNN_OPS_EVENT], [PYDTNN_EVENT_FINISHED, PYDTNN_EVENT_FINISHED])

    def _train_batch(self, x_batch, y_batch, current_batch_size, sync_model=True):
        self.mode = TRAIN_MODE

        # LR schedulers begin
        for lr_sched in self.lr_schedulers:
            lr_sched.on_batch_begin()

        try:
            x, y_targ = self._get_x_y_targ(x_batch, y_batch, current_batch_size)
        except ValueError:
            return self.total_metrics

        if x_batch.shape[0] > 0:
            # Forward pass (FP)
            for i in range(1, len(self.layers)):
                self.tracer.emit_event(PYDTNN_MDL_EVENT, self.layers[i].id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.FORWARD.value)
                x = self.layers[i].forward(x)
                self.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)

            loss, dx = self.loss_func(x, y_targ, self.batch_size)
        else:
            assert y_targ.shape[0] == 0
            loss, dx = 0.0, y_targ

        self.total_metrics, _ = self._compute_metrics_funcs(x, y_targ, loss, comm=sync_model)

        if x_batch.shape[0] > 0:
            # Backward pass (BP)
            for i in range(len(self.layers) - 1, 0, -1):
                self.tracer.emit_event(PYDTNN_MDL_EVENT, self.layers[i].id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.BACKWARD.value)
                dx = self.layers[i].backward(dx)
                self.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)

        if self.enable_cudnn:
            self.stream.synchronize()

        # Gradient update
        if sync_model:
            self._weight_update(gradient=True, blocking=self.blocking_mpi)

        if x_batch.shape[0] > 0 or sync_model:

            # Optimizer
            for i in range(len(self.layers) - 1, 0, -1):
                self.tracer.emit_event(PYDTNN_MDL_EVENT, self.layers[i].id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.UPDATE_DW.value)
                self.layers[i].update_weights(self.optimizer)
                self.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)

        # Weight update
        if self.model_sync_freq > 0 and sync_model:
            self._weight_update(gradient=False, blocking=self.blocking_mpi)

        if self.enable_cudnn:
            for i in range(len(self.layers) - 1, 0, -1):
                if self.layers[i].grad_vars:
                    self.layers[i].stream_2.synchronize()

        # LR schedulers end
        for lr_sched in self.lr_schedulers:
            lr_sched.on_batch_end(self)

        return self.total_metrics

    def _initialize(self):
        if self._initialized:
            return
        self._apply_layer_fusion(self.kwargs.get("enable_fused_bn_relu"), self.kwargs.get("enable_fused_conv_relu"),
                                 self.kwargs.get("enable_fused_conv_bn"), self.kwargs.get("enable_fused_conv_bn_relu"))
        self.loss_func = getattr(losses, self.loss_func_name)(shape=(self.batch_size, *self.layers[-1].shape),
                                                              model=self)
        self.metrics_funcs = [getattr(metrics, m)(shape=(self.batch_size, *self.layers[-1].shape), model=self) for m in
                              self.metrics_list]
        self.loss_and_metrics = [self.loss_func_name] + self.metrics_list
        self.total_metrics = np.array([0] + [0 for func in self.metrics_funcs], dtype=self.dtype)
        self.tracer.define_event_types(self)
        self._initialized = True

    def train(self, x_train, y_train, x_val, y_val, bar_width=BAR_WIDTH):
        self.dataset = CustomDataset(self, x_train=x_train, y_train=y_train, x_test=x_val, y_test=y_val)
        history = self.train_dataset(bar_width=bar_width)
        return history

    def _compute_rank_weight(self, mask):
        match self.model_sync_participation:
            case "all":
                comm_nsamples = self.comm_nsamples
            case "avail2all":
                if mask[self.rank]:
                    comm_nsamples = [nsamples for nsamples, mask in zip(self.comm_nsamples, mask) if mask]
                else:
                    return 0.0
            case _:
                raise NotImplementedError(f"model_sync_participation = {self.model_sync_participation}")

        min_nsamples, max_nsamples, total_nsamples = min(comm_nsamples), max(comm_nsamples), sum(comm_nsamples)
        comm_size = len(comm_nsamples)

        match self.model_sync_alg:
            case "avg":
                return 1.0 / comm_size
            case "wavg":
                return self.dataset.train_nsamples / total_nsamples
            case "invwavg":
                inverse_nsamples = min_nsamples + (max_nsamples - self.dataset.train_nsamples)
                return inverse_nsamples / self.total_nsamples

    @ensure_model_is_initialized
    def train_dataset(self, bar_width=BAR_WIDTH):
        if self.enable_cudnn and self.y_batch is None:
            self.y_batch = pydtnn.backends.gpu.tensor_gpu.TensorGPU(
                gpuarray.empty((self.batch_size, *self.layers[-1].shape), self.dtype),
                self.tensor_format, self.cudnn_dtype)

        self.history = {lm: [] for lm in (self.loss_and_metrics + [f"val_{m}" for m in self.loss_and_metrics])}

        terminate = False

        model_sync_count = 0
        for epoch in range(self.num_epochs):
            train_batch_generator, val_batch_generator = self.dataset.get_train_val_generator()

            train_total_loss, train_batch_count = np.zeros(len(self.loss_and_metrics)), 0
            val_total_loss, val_batch_count = np.zeros(len(self.loss_and_metrics)), 0

            if self.comm_rank == 0:
                string = ""
                fmt = "%%%dd" % (len(str(self.num_epochs)))
                epoch_string = "Epoch %s/%s" % (fmt, fmt)
                pbar = tqdm(total=self.dataset.train_nsamples, ncols=bar_width,
                            ascii=" ▁▂▃▄▅▆▇█", smoothing=0.3,
                            desc=epoch_string % (epoch + 1, self.num_epochs), unit=" samples")

            for lr_sched in self.lr_schedulers:
                lr_sched.on_epoch_begin(self, self.rank)

            for x_batch, y_batch, batch_size in train_batch_generator:
                sync_model = (self.model_sync_freq <= 0) or (model_sync_count % self.model_sync_freq == 0)
                model_sync_count += 1

                rank_mask = self.comm.allgather(min(1, batch_size)) if self.comm else [min(1, batch_size)]
                rank_avail = sum(rank_mask)

                if rank_avail <= 0:
                    break

                if rank_avail < self.model_sync_min_avail:
                    sync_model = False

                self.rank_weight = self._compute_rank_weight(rank_mask)

                tic = timer()
                train_batch_loss = self._train_batch(x_batch, y_batch, batch_size, sync_model=sync_model)
                toc = timer()

                if batch_size <= 0:
                    if self.comm_rank == 0:
                        pbar.set_postfix_str(s=f"{string}, waiting…", refresh=True)
                    continue

                train_total_loss, train_batch_count, string = \
                    self._update_running_average(train_batch_loss, train_total_loss,
                                                 train_batch_count, batch_size)
                if self.comm_rank == 0:
                    # noinspection PyUnboundLocalVariable
                    pbar.set_postfix_str(s=string, refresh=True)
                    pbar.update(batch_size)
                    self.perf_counter.add_training_time_and_batch_size(epoch, toc - tic, batch_size)

            if self.comm_rank == 0:
                train_string = string
                for c in range(len(self.loss_and_metrics)):
                    self.history[self.loss_and_metrics[c]].append(train_total_loss[c])

            for x_batch, y_batch, batch_size in val_batch_generator:
                sync_model = (self.model_sync_freq <= 0) or (model_sync_count % self.model_sync_freq == 0)
                model_sync_count += 1

                rank_mask = self.comm.allgather(min(1, batch_size)) if self.comm else [min(1, batch_size)]
                rank_avail = sum(rank_mask)

                if rank_avail <= 0:
                    break

                if rank_avail < self.model_sync_min_avail:
                    sync_model = False

                val_batch_loss = self._evaluate_batch(x_batch, y_batch, batch_size, sync_model=False and sync_model)

                if batch_size <= 0:
                    continue

                val_total_loss, val_batch_count, string = \
                    self._update_running_average(val_batch_loss, val_total_loss,
                                                 val_batch_count, batch_size, prefix="val_")
                if self.comm_rank == 0:
                    pbar.set_postfix_str(s=f"{train_string}, {string}", refresh=True)

            if self.comm_rank == 0:
                for c in range(len(self.loss_and_metrics)):
                    self.history["val_" + self.loss_and_metrics[c]].append(val_total_loss[c])

            for lr_sched in self.lr_schedulers:
                lr_sched.on_epoch_end(train_total_loss, val_total_loss)
                if getattr(lr_sched, "stop_training", False):
                    terminate = True

            if self.comm_rank == 0:
                pbar.close()
                # Sleep for half a second to allow pbar to write its output before returning
                time.sleep(.5)

            if terminate:
                break

        # Syncronize model
        if self.final_model_sync:
            self._weight_update(gradient=True, blocking=self.blocking_mpi)
            self._weight_update(gradient=False, blocking=self.blocking_mpi)

        self.tracer.define_event_types(self)
        return self.history

    def _evaluate_batch(self, x_batch, y_batch, current_batch_size, sync_model=True):
        self.mode = EVALUATE_MODE

        try:
            x, y_targ = self._get_x_y_targ(x_batch, y_batch, current_batch_size)
        except ValueError:
            return self.total_metrics

        # Forward pass (FP)
        if x_batch.shape[0] > 0:
            for i in range(1, len(self.layers)):
                self.tracer.emit_event(PYDTNN_MDL_EVENT, self.layers[i].id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.FORWARD.value)
                x = self.layers[i].forward(x)
                self.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)

            y_pred = self.layers[-1].y
            loss, _ = self.loss_func(y_pred, y_targ, self.batch_size)
        else:
            y_pred = self.layers[-1].y
            loss = 0.0

        self.total_metrics, _ = self._compute_metrics_funcs(y_pred, y_targ, loss, comm=sync_model)

        return self.total_metrics

    def evaluate(self, x_test, y_test, bar_width=BAR_WIDTH):
        self.dataset = CustomDataset(self, x_test=x_test, y_test=y_test)
        self.evaluate_dataset(bar_width=bar_width)

    @ensure_model_is_initialized
    def evaluate_dataset(self, bar_width=BAR_WIDTH):
        if self.enable_cudnn and self.y_batch is None:
            self.y_batch = pydtnn.backends.gpu.tensor_gpu.TensorGPU(
                gpuarray.empty((self.batch_size, *self.layers[-1].shape), self.dtype),
                self.tensor_format, self.cudnn_dtype)

        test_batch_generator = self.dataset.get_test_generator()

        if self.comm_rank == 0:
            test_total_loss, test_batch_count = np.zeros(len(self.loss_and_metrics)), 0
            pbar = tqdm(total=self.dataset.test_nsamples, ncols=bar_width,
                        ascii=" ▁▂▃▄▅▆▇█", smoothing=0.3,
                        desc="Testing", unit=" samples")

        model_sync_count = 0
        for x_batch, y_batch, batch_size in test_batch_generator:
            sync_model = (self.model_sync_freq <= 0) or (model_sync_count % self.model_sync_freq == 0)
            model_sync_count += 1

            rank_mask = self.comm.allgather(min(1, batch_size)) if self.comm else [min(1, batch_size)]
            rank_avail = sum(rank_mask)

            if rank_avail <= 0:
                break

            if rank_avail < self.model_sync_min_avail:
                sync_model = False

            tic = timer()
            test_batch_loss = self._evaluate_batch(x_batch, y_batch, batch_size, sync_model=sync_model)
            toc = timer()

            if batch_size <= 0:
                continue

            if self.comm_rank == 0:
                # noinspection PyUnboundLocalVariable
                test_total_loss, test_batch_count, string = \
                    self._update_running_average(test_batch_loss, test_total_loss, test_batch_count, batch_size,
                                                 prefix="test_")
                # noinspection PyUnboundLocalVariable
                pbar.set_postfix_str(s=string, refresh=True)
                pbar.update(batch_size)
                self.perf_counter.add_testing_time_and_batch_size(self._evaluate_round, toc - tic, batch_size)

        # Increment self._evaluate_round
        self._evaluate_round += 1

        if self.comm_rank == 0:
            pbar.close()
            # Sleep for half a second to allow pbar to write its output before returning
            time.sleep(.5)
