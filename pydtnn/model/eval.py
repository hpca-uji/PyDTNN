"""
Evaluation module for PyDTNN models.

This module provides the Eval class, which handles model evaluation,
metric computation, and performance tracking during the testing phase.
"""

import logging
import time
from timeit import default_timer as timer
from typing import Any, Generator

import numpy as np
from tqdm import tqdm

from pydtnn import MPI, gpuarray
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.datasets.abstract import Dataset
from pydtnn.layers.input import Input
from pydtnn.model.base import Base
from pydtnn.model.sync import Sync
from pydtnn.tracers.events import (PYDTNN_EVENT_FINISHED, PYDTNN_MDL_EVENT,
                                   PYDTNN_MDL_EVENTS, MdlEventEnum)
from pydtnn.utils import TqdmLogger
from pydtnn.utils.constants import Array
from pydtnn.utils.performance_models import allreduce_time

__all__ = ("Eval",)

logger = logging.getLogger(__name__)


class Eval[T: Array](Sync[T]):
    """
    Handles the evaluation logic for distributed models.

    Extends Sync to provide evaluation-specific synchronization and
    metric aggregation across distributed processes.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initializes the Eval instance."""
        super().__init__(**kwargs)
        # Private attributes
        self._evaluate_round: int = 0

    def _compute_metrics_funcs(self, y_pred: T, y_targ: T, loss: float, blocking: bool = True,
                               comm: bool = True) -> tuple[np.ndarray, None] | tuple[None, Any]:
        """
        Computes metrics and loss, optionally synchronizing across processes.

        Args:
            y_pred: Predicted output tensor.
            y_targ: Target output tensor.
            loss: Calculated loss value.
            blocking: Whether to use blocking MPI communication.
            comm: Whether to perform MPI communication.

        Returns:
            A tuple containing the aggregated metrics/loss and an optional MPI request.
        """
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

        return _losses, loss_req  # type: ignore

    def _update_running_average(
        self,
        batch_metric: np.ndarray,
        total_metric: np.ndarray,
        total_size: int,
        batch_size: int,
        prefix: str = "",
    ) -> tuple[np.ndarray, int, str]:
        """
        Updates the running average of metrics and generates a status string.

        Args:
            curr: Current batch metrics.
            total: Accumulated metrics.
            count: Total samples processed.
            batch_size: Current batch size.
            prefix: String prefix for output.

        Returns:
            Updated total metrics, updated count, and formatted status string.
        """
        string = ""
        total_metric = ((batch_metric * batch_size) + (total_metric * total_size)) / (
            total_size + batch_size
        )
        total_size += batch_size
        for c in range(len(self.loss_and_metrics)):
            loss_str = self.loss_and_metrics_format[c]
            if loss_str:
                string += ("%s, " % (prefix + loss_str)) % total_metric[c]
        string = string[:-2]
        return total_metric, total_size, string

    def _evaluate_batch(
        self, x_batch: np.ndarray, y_batch: np.ndarray, sync_model: bool = True
    ) -> np.ndarray:
        """
        Performs a forward pass and metric computation for a single batch.

        Args:
            x_batch: Input data batch.
            y_batch: Target data batch.
            sync_model: Whether to synchronize metrics across processes.

        Returns:
            The computed metrics for the batch.
        """
        self.mode = Base.Mode.EVALUATE

        self.real_batch_size = x_batch.shape[0]
        input_layer: Input[T] = self.layers[0]  # type: ignore (casting to the right type)
        x, y_targ = input_layer._sync_x_y(x_batch, y_batch)

        has_batch = x_batch.shape[0] > 0

        # Forward pass (FP)
        if has_batch:
            for i in range(len(self.layers)):
                self.tracer.emit_event(
                    PYDTNN_MDL_EVENT,
                    self.layers[i].id * PYDTNN_MDL_EVENTS + MdlEventEnum.FORWARD,
                )
                x = self.layers[i].forward(x)
                self.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)
            loss, _ = self.loss_func.compute(x, y_targ, self.real_batch_size)
        else:
            if y_targ.shape[0] != x_batch.shape[0]:
                raise ValueError(
                    f"y_targ.shape[0] ({y_targ.shape[0]})"
                    f" and x_batch.shape[0] ({x_batch.shape[0]})"
                    " must have the same value."
                )
            loss, _ = 0.0, y_targ

        total_metrics = None
        total_metrics, _ = self._compute_metrics_funcs(x, y_targ, loss, comm=sync_model)
        assert total_metrics is not None
        self.total_metrics = total_metrics

        return self.total_metrics

    def _update_status(
        self,
        pbar: tqdm | None,
        batch_loss: np.ndarray,
        total_loss: np.ndarray,
        total_size: int,
        batch_size: int,
        output_prefix: str,
        current_round: int,
        delta: float = -1,
        prev_string: str = "",
    ) -> tuple[np.ndarray, int, str]:
        """
        Updates the progress bar and internal performance counters.

        Args:
            pbar: Tqdm progress bar instance.
            batch_loss: Loss/metrics for the current batch.
            total_loss: Accumulated loss/metrics.
            total_size: Total samples processed.
            batch_size: Size of the current batch.
            output_prefix: Prefix for logging.
            current_round: The current training/evaluation round.
            delta: Time taken for the batch.
            prev_string: Previous status string.

        Returns:
            Updated total metrics, updated count, and formatted status string.
        """

        part = Dataset.Part[output_prefix.strip("_").upper()]

        # noinspection PyUnboundLocalVariable
        total_loss, total_size, string = self._update_running_average(
            batch_metric=batch_loss,
            total_metric=total_loss,
            total_size=total_size,
            batch_size=batch_size,
            prefix=output_prefix,
        )

        self.perf_counter._add_time_and_batch_size(part, current_round, delta, batch_size)

        if self.comm_rank == 0:
            # noinspection PyUnboundLocalVariable
            # type: ignore (pbar is a 'tqdm', it only is None in self.comm_rank != 0)
            pbar.set_postfix_str(s=f"{prev_string}{string}", refresh=True)
            if part != Dataset.Part.VAL:
                # type: ignore (Here there is a 'tqdm' object, pbar only is None in
                # self.comm_rank != 0)
                pbar.update(batch_size)

        return total_loss, total_size, string

    def _evalutate_round(
        self,
        pbar: tqdm | None,
        batch_generator: Generator[tuple[np.ndarray, np.ndarray, int]],
        model_sync_count: int,
        batches_min: float,
        total_loss: np.ndarray,
        total_size: int,
        terminate: bool = False,
        prev_string: str = "",
        out_prefix: str = "",
    ) -> tuple[np.ndarray, int, int, bool, str]:
        """
        Executes a single evaluation round over the provided batch generator.

        Returns:
            A tuple containing updated total loss, sync count, sync status, and status string.
        """
        sync_epoch = False
        string = ""
        part = Dataset.Part[out_prefix.rstrip("_").upper()]

        for i_batch, (x_batch, y_batch, batch_size) in enumerate(batch_generator):
            if terminate:
                x_batch = x_batch[:0]
                y_batch = y_batch[:0]
            local_batch_size = x_batch.shape[0]

            sync_model = (self.model_sync_freq <= 0) or (
                model_sync_count % self.model_sync_freq == 0
            )

            if sync_model:
                sync_epoch = True

            if model_sync_count == 0 and not self.initial_model_sync:
                sync_model = False

            model_sync_count += 1

            if i_batch < batches_min:
                rank_mask = [1] * self.comm_size
            else:
                rank_mask = (
                    self.comm.allgather(min(1, local_batch_size))
                    if self.comm
                    else [min(1, local_batch_size)]
                )
            rank_avail = sum(rank_mask)

            if rank_avail <= 0:
                break

            if rank_avail < self.model_sync_min_avail:
                sync_model = False

            self.rank_weight = self._compute_rank_weight(rank_mask, part)

            tic = timer()
            test_batch_loss = self._evaluate_batch(x_batch, y_batch, sync_model=sync_model)
            toc = timer()
            delta = toc - tic

            if part is not Dataset.Part.TEST:
                delta = -1

            if batch_size <= 0:
                continue

            total_loss, total_size, string = self._update_status(
                pbar=pbar,
                batch_loss=test_batch_loss,
                total_loss=total_loss,
                total_size=total_size,
                batch_size=batch_size,
                output_prefix=out_prefix,
                current_round=self._evaluate_round,
                delta=delta,
                prev_string=prev_string,
            )

        # Increment self._evaluate_round
        self._evaluate_round += 1
        return (total_loss, total_size, model_sync_count, sync_epoch, string)

    def evaluate(self) -> None:
        """
        Runs the full evaluation process on the test dataset.

        Args:
            bar_width: Width of the progress bar.
        """
        self._ensure_model_runnable()

        if self.enable_cudnn and self.y_batch is None:
            assert gpuarray and self.cudnn_dtype
            tensor_ary = TensorArray(
                gpuarray.empty((self.batch_size, *self.layers[-1].shape), self.dtype),
                self.tensor_format,
                self.cudnn_dtype,
            )
            self.y_batch = tensor_ary  # type: ignore

        self.comm_nsamples = list(
            zip(
                *(
                    self.comm.allgather(self.dataset._nsamples)
                    if self.comm
                    else [self.dataset._nsamples]
                )
            )
        )

        test_batch_generator = self.dataset.get_test_generator()
        test_batches_min: float = min(self.comm_nsamples[Dataset.Part.TEST]) / (
            self.batch_size * self.nprocs
        )
        test_total_loss, test_total_size = np.zeros(len(self.loss_and_metrics), np.float32), 0

        if self.comm_rank == 0:
            pbar = tqdm(
                file=TqdmLogger(),
                total=self.dataset.test_nsamples,
                ascii=" ▁▂▃▄▅▆▇█",
                smoothing=0.3,
                desc="Testing",
                unit=" samples",
            )
        else:
            pbar = None

        self._evalutate_round(
            pbar=pbar,
            batch_generator=test_batch_generator,
            model_sync_count=0,
            batches_min=test_batches_min,
            total_loss=test_total_loss,
            total_size=test_total_size,
            out_prefix=f"{Dataset.Part.TEST._name_.lower()}_",
        )

        if self.comm_rank == 0:
            pbar.close()  # type: ignore (Here is a 'tqdm', only is None in self.comm_rank != 0)
            # Sleep for half a second to allow pbar to write its output before returning
            time.sleep(0.5)

        # End pipelines
        self._model_reduce_wait(gradient=True)
        self._model_reduce_wait(gradient=False)

    def calculate_time(self) -> np.ndarray:
        """
        Calculates the estimated time for various model operations.

        Returns:
            A numpy array containing total, computation, memory, and network time estimates.
        """
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
                    total_time += allreduce_time(
                        weights_size + biases_size,
                        self.cpu_speed,
                        self.network_bw,
                        self.network_lat,
                        self.network_algo,
                        self.nprocs,
                        self.dtype,
                    )
        else:
            total_time_iar: int = 0
            # Non-blocking MPI
            # Back propagation. Gradient computation (GC) and weights update (WU)
            for layer in self.layers:
                total_time += layer.bwd_time
                weights_size = 0 if (weights := layer.weights) is None else weights.size
                biases_size = 0 if (biases := layer.biases) is None else biases.size
                if self.comm and weights_size > 0:
                    time_iar = allreduce_time(
                        weights_size + biases_size,
                        self.cpu_speed,
                        self.network_bw,
                        self.network_lat,
                        self.network_algo,
                        self.nprocs,
                        self.dtype,
                    )
                    total_time[3] += time_iar[3]
                    total_time_iar = max(total_time[0], total_time_iar) + time_iar[0]

            total_time[0] = max(total_time[0], total_time_iar)

        return total_time
