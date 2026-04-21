


from warnings import warn

from pydtnn._model.model_base import Model_Base
from pydtnn.abstract.layerable import Layerable
from pydtnn.tracers.extrae_tracer import ExtraeTracer
from pydtnn.tracers.simple_tracer import SimpleTracer
from pydtnn.tracers.simple_tracer_gpu import SimpleTracerPycuda
from pydtnn.tracers.simple_tracer_pmlib import SimpleTracerPMLib
from pydtnn.utils.constants import Array
from pydtnn.utils.tensor import TensorFormat


class Model_Utils[T: Array](Model_Base[T]):
    def _update_running_average(self, curr: np.ndarray, total: np.ndarray, count: int,
                                batch_size: int, prefix="") -> tuple[np.ndarray, int, str]:
        string = ""
        total = ((curr * batch_size) + (total * count)) / (count + batch_size)
        for c in range(len(self.loss_and_metrics)):
            loss_str = self.loss_and_metrics_format[c]
            if loss_str:
                string += ("%s, " % (prefix + loss_str)) % total[c]
        string = string[:-2]
        return total, count + batch_size, string


    def _compute_metrics_funcs(self, y_pred: T, y_targ: T, loss: float, blocking=True, comm=True) -> tuple[np.ndarray, None] | tuple[None, Any]:
        loss_req: Any | None = None
        _losses: np.ndarray | None

        if y_targ.shape[0] > 0:
            metrics = [func.compute(y_pred, y_targ) for func in self.metrics_funcs]
            _losses = np.array([loss, *metrics], dtype=np.object_)
        else:
            _losses = self.total_metrics.copy()
            _losses[0] = loss

        if self.comm is not None and comm:
            assert MPI

            _losses /= self.comm_size
            if blocking:
                _losses = self.comm.allreduce(_losses, op=MPI.SUM)
            else:
                loss_req = self.comm.iallreduce(_losses, op=MPI.SUM)
        else:
            if blocking:
                pass
            else:
                raise NotImplementedError("can not compute metrics non-blocking locally")

        return _losses, loss_req
    # ----


    def calculate_time(self) -> np.ndarray:
        # Total elapsed_time, Comp elapsed_time, Memo elapsed_time, Net elapsed_time
        total_time: np.ndarray = np.zeros((4,), dtype=np.float32)

        # Forward pass (FP)
        for layer in self.layers:
            total_time += layer.fwd_time

        if self.blocking_mpi:
            # Blocking MPI
            # Back propagation. Gradient computation (GC) and weights update (WU)
            for layer in self.layers:
                total_time += layer.bwd_time

            # Weight update (WU)
            for layer in self.layers:
                weights_size = 0 if (weights := layer.weights) is None else weights.size
                biases_size = 0 if (biases := layer.biases) is None else biases.size
                if self.comm and weights_size > 0:
                    total_time += allreduce_time(weights_size + biases_size,
                                                 self.cpu_speed, self.network_bw, self.network_lat,
                                                 self.network_alg, self.nprocs, self.dtype)
        else:
            total_time_iar: int = 0
            # Non-blocking MPI
            # Back propagation. Gradient computation (GC) and weights update (WU)
            for layer in self.layers:
                total_time += layer.bwd_time
                weights_size = 0 if (weights := layer.weights) is None else weights.size
                biases_size = 0 if (biases := layer.biases) is None else biases.size
                if self.comm and weights_size > 0:
                    time_iar = allreduce_time(weights_size + biases_size,
                                              self.cpu_speed, self.network_bw, self.network_lat,
                                              self.network_alg, self.nprocs, self.dtype)
                    total_time[3] += time_iar[3]
                    total_time_iar = max(total_time[0], total_time_iar) + time_iar[0]

            total_time[0] = max(total_time[0], total_time_iar)

        return total_time

    from pydtnn.models.model import select as select_model
    def _read_model(self, model_name: str) -> None:
        create_model = select_model(model_name)

        # NOTE: Dataset is always in NCHW
        # Change input_shape to model.tensor_format
        input_shape = format_reshape(self.dataset.input_shape, SampleFormat.CHW, self.tensor_format.as_sample())  # type: ignore
        if len(input_shape) != 3:
            warn_text = f"Input layer does not have 3 dimensions ({input_shape}), it may cause issues!"
            logger.warning(warn_text)
            warn(warn_text, RuntimeWarning)
        launch_shape_warning = len(input_shape) == 3 and not (input_shape[0] > input_shape[2]) if self.tensor_format is TensorFormat.NHWC \
            else len(input_shape) == 3 and not (input_shape[0] < input_shape[1])
        if launch_shape_warning:
            warn_text = f"Input layer shape {input_shape} may not be in {self.tensor_format} format, regardless of model format! "
            logger.warning(warn_text)
            warn(warn_text, RuntimeWarning)
        output_shape = tuple(self.dataset.output_shape)

        layers = create_model(input_shape, output_shape)
        self.add_layers(layers)  # type: ignore


    def get_all_layers(self, from_layers: list[Layerable[T]] | None = None) -> list[Layerable[T]]:
        if from_layers is None:
            from_layers = self.layers
        this_recursion_layers = []
        for layer in from_layers:
            this_recursion_layers.append(layer)
            children = layer.children
            this_recursion_layers += self.get_all_layers(children)
        return this_recursion_layers
    


def get_tracer(tracer_output: str, tracing: bool, comm: MPI_COMM | None, enable_cudnn: bool,
               tracer_pmlib_server: str, tracer_pmlib_port: int, tracer_pmlib_device: str) -> Tracer:

    if tracer_output == "":
        tracer = ExtraeTracer(tracing)
    else:
        if enable_cudnn:
            tracer = SimpleTracerPycuda(tracing, tracer_output, comm)
        else:
            if tracer_pmlib_device != "":
                tracer = SimpleTracerPMLib(tracing, tracer_output, comm, tracer_pmlib_server, tracer_pmlib_port, tracer_pmlib_device)
            else:
                tracer = SimpleTracer(tracing, tracer_output, comm)
    return tracer


def get_tensor_format(tensor_format: Literal["AUTO", "NCHW", "NHWC"] = "AUTO", gpu: bool = False) -> TensorFormat:
    match tensor_format.upper():
        case "AUTO":
            return TensorFormat.NCHW if gpu else TensorFormat.NHWC
        case "NCHW":
            return TensorFormat.NCHW
        case "NHWC":
            return TensorFormat.NHWC
        case _:
            raise NotImplementedError(f"\'{tensor_format}\' is not supported.")


def get_batch_size(local_size: int | None, global_size: int | None, comm_size: int, default: int = DEFAULT_BACH_SIZE) -> int:
    if local_size and global_size:
        raise ValueError("Can not define 'local_batch_size' and 'global_batch_size' simultaneously")

    if global_size:
        # NOTE: Using comm_size instead of nprocs might not be appropriate,
        #       as it differs to how global_batch_size is defined elsewhere,
        #       but for now it just a parser option difference that helps testing
        batch_size = global_size // comm_size
    elif local_size:
        batch_size = local_size
    else:
        batch_size = default

    if batch_size < 1:
        raise ValueError(f"'batch_size' ({batch_size}) too small or too many processes (num processes: {comm_size})")

    return batch_size
