import enum
from typing import Any
from warnings import warn
import numpy as np
import logging

logger = logging.getLogger(__name__)

from pydtnn.parser import PydtnnArgumentParser
from pydtnn.utils.constants import Array, NetworkAlgEnum, Parameters

class Model_Base[T: Array]:

    """
    PyDTNN Model
    """
    class Mode(enum.StrEnum):
        EVALUATE = enum.auto()
        TRAIN = enum.auto()
        UNSPECIFIED = enum.auto()

# Explicit declaration of those model attributes that are referenced by other parts of PyDTNN
#   NOTE: The following parameters come from "Parser"
    steps_per_epoch: int
    cpu_speed: float
    memory_bw: float
    network_bw: float
    network_lat: float
    network_alg: NetworkAlgEnum
    loss_func_name: str
    num_epochs: int
    model_sync_freq: int
    final_model_sync: bool
    test_as_validation: bool
    validation_split: float
    use_synthetic_data: bool
    dataset_train_path: str
    dataset_test_path: str
    evaluate_only: bool
    evaluate_on_train: bool
    profile: bool
    history_file: str
    model_sync_min_avail: int
    dataset_name: str
    shared_storage: bool
    encryption_name: str
    augment_flip: float
    augment_crop_size: int
    augment_crop: float
    transform_crop: bool
    transform_crop_perc: int
    transform_resize: bool
    transform_resize_dsize: int
    initial_model_sync: bool
    dataset_percentage: float
    use_mpi_buffers: bool
    # enable_memory_cache: bool
    gpus_per_node: int
    weights_and_bias_filename: str
    learning_rate_scaling: bool
    metrics: str
    use_memory_pool: bool
    augment_shuffle: bool
    normalize: bool
    transform_resize_size: int
    normalize_offset: float
    normalize_scale: float
    model_sync_quantize: bool
    model_sync_dtype: np.dtype
# ------------

    rank_weight: float
    comm_rank: int
    comm_size: int
    rank: int
    nprocs: int
    learning_rate: float
    MPI: MPI_MODULE | None
    comm: MPI_COMM | None

    nccl_type: Any | None
    nccl_comm: Any | None
    cudnn_handle: Cudnn_Handle_Type | None
    cublas_handle: Cublas_Handle_Type | None
    stream: Any  # drv.Stream
    cudnn_dtype: int
    input_shape: ArrayShape
    output_shape: ArrayShape

    dtype: np.dtype

    real_batch_size: int
    nparams: int

    y_batch: T
    history: dict[str, list[np.ndarray]]
    loss_func: Loss
    metrics_funcs: list[Metric]
    loss_and_metrics: list[str]
    total_metrics: np.ndarray

    cuda_grid: tuple[int, int, int]
    cuda_block: tuple[int, int, int]
    
##########################################
    ## INIT ##
    ##########
    def __init__(self, **kwargs):

        # Get default values from parser and update them from the received kwargs
        self.kwargs: dict[str, Any] = PydtnnArgumentParser().get_default_values()
        self.kwargs.update(kwargs)

        # Attributes related to the given arguments
        self.blocking_mpi: bool = self.use_blocking_mpi
        self.enable_cudnn = gpuarray is not None and drv is not None and cublas is not None
        self.gpudirect: bool = self.enable_gpudirect
        self.enable_nccl: bool = self.enable_nccl
        self.dtype: np.dtype = np.dtype(self.dtype)
        self.memory: PrivateMemory = None  # type: ignore (it will be intialized later if "self.use_memory_pool" is True)

        self.nparams = 0
        self.memory_used = 0
        self.tmp_memory_used = 0

        # Set MPI and comm
        self._mpi_init()

        # Set performance counter
        self.perf_counter = PerformanceCounter()

        # Layers' attributes
        self.layers: list[Layerable] = []
        self.layer_id_generator: abc.Iterator[int] = iter(itertools.count())

        # Set current mode to unspecified
        self.mode: Model.Mode = Model.Mode.UNSPECIFIED

        # Memory cache optimization
        # if self.enable_memory_cache:
        #     MemoryCache.enable()
        # else:
        #     MemoryCache.disable()

        self.memory_cls = PreallocMemory if self.shared_tmp_memory else PrivateMemory

        # Set tracer
        self.tracer = get_tracer(tracer_output=self.tracer_output, tracing=self.tracing, comm=self.comm, enable_cudnn=self.enable_cudnn,
                                 tracer_pmlib_server=self.tracer_pmlib_server, tracer_pmlib_port=self.tracer_pmlib_port,
                                 tracer_pmlib_device=self.tracer_pmlib_device)

        # Cuda
        if self.enable_cudnn:
            self._cudnn_init()

        # Data format
        self.tensor_format: TensorFormat = get_tensor_format(tensor_format=self.tensor_format, gpu=self.enable_cudnn)  # type: ignore

        # Disable BestOf globally if not enabled
        # if self.enable_best_of is False:
        #     BestOf.use_always_the_first_alternative()

        self.batch_size = get_batch_size(local_size=self.batch_size, global_size=self.global_batch_size, comm_size=self.comm_size)

        # Attributes that will be properly initialized elsewhere

        # ---

        # Encryption
        if self.encryption_name:
            self.crypt = self._crypt_init(self.encryption_name)

        else:
            self.crypt = None

        # Load weights and bias
        if self.weights_and_bias_filename:
            self.load_weights_and_bias(self.weights_and_bias_filename)
        # Dataset
        if self.dataset_name:
            self.dataset: Dataset = select_dataset(self.dataset_name)(self)

        # Optimizers and LRSchedulers
        if self.learning_rate_scaling:
            # using comm_size instead of nprocs might not be appropriate,
            # as it differs to how learning_rate is defined elsewhere,
            # but for now it just a parser option difference that helps testing
            self.learning_rate = self.learning_rate / self.comm_size

        self.optimizer = select_optimizer(self.optimizer_name).from_model(self)
        self.optimizer._init_backend_with_model(self)

        self.schedulers = [
            select_scheduler(scheduler_name).from_model(self)
            for scheduler_name in filter(None, self.schedulers_names.split(","))
        ]
        for scheduler in self.schedulers:
            scheduler.model = self

        # Metrics list
        self.metrics_list: list[str] = [m for m in self.metrics.replace(" ", "").split(",")]

        # Private attributes
        self._evaluate_round: int = 0
        self._is_model_init: bool = False

        # Read the model (must be the last action, as it calls self._model_init() if there is a model)
        self.model_name: str | None = self.kwargs.get("model_name")
        if self.model_name:
            self._read_model(self.model_name)


    
    
    def export(self) -> dict[str, Any]:
        data = {}

        if self.model_name is not None:
            data[Parameters.MODEL_NAME] = self.model_name

        data[Parameters.LAYERS] = [
            layer.export()
            for layer in self.layers
        ]

        return data

    def import_(self, data: "dict[str, Any] | Model") -> None:
        if isinstance(data, Model_Base):
            data = data.export()

        model_name = str(data.get(Parameters.MODEL_NAME))
        if model_name != self.model_name:
            warn_text = f"Importing from different models! (self: {self.model_name}, got: {model_name})"
            logger.warning(warn_text)
            warn(warn_text, RuntimeWarning)

        for layer, data in zip(self.layers, data[Parameters.LAYERS]):
            layer.import_(data)  # type: ignore (It is the right data type.)

    def load_weights_and_bias(self, filename: str) -> None:
        """
        ARGS:
            filename: Path to the file with the weights and biases to load.
        """
        with np.load(filename, allow_pickle=True) as data:
            self.import_(data)

    def store_weights_and_bias(self, filename: str, compress=True) -> None:
        """
        ARGS:
            filename: Path to the file were the weights and biases will be stored.
        """
        save = np.savez_compressed if compress else np.savez
        save(filename, **self.export())