"""
Evaluation module for PyDTNN models.

This module provides the Eval class, which handles model evaluation,
metric computation, and performance tracking during the testing phase.
"""

import logging
import time
from timeit import default_timer as timer
from typing import Any
from collections.abc import Generator

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
from pydtnn.utils.constants import Array
from pydtnn.utils.logs import TqdmLogger
from pydtnn.utils.performance_models import allreduce_time

__all__ = ("Eval",)

logger = logging.getLogger(__name__)


class Eval[T: Array](Sync[T]):  # noqa: D101 (generics not detected)
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

    def _compute_metrics_funcs(
        self, y_pred: T, y_targ: T, loss: float, batch_size: int
    ) -> tuple[np.ndarray, None] | tuple[None, MPI.Request]:
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
        loss_req: MPI.Request | None = None
        _losses: np.ndarray | None

        if batch_size > 0:
            metrics = [func.compute(y_pred, y_targ) for func in self.metrics_funcs]
            _losses = np.array([loss, *metrics, 1], dtype=np.object_)
            _losses *= batch_size
        else:
            _losses = np.zeros(len(self.metrics_funcs) + 2, dtype=np.object_)

        return _losses, loss_req  # pyright: ignore[reportReturnType]

    def _format_metrics(
        self,
        metric: np.ndarray,
        prefix: str = "",
    ) -> str:
        """Generates a metrics status string"""
        string = ""
        for c in range(len(self.loss_and_metrics)):
            loss_str = self.loss_and_metrics_format[c]
            if loss_str:
                string += ("%s, " % (prefix + loss_str)) % (metric[c] / metric[-1])
        string = string[:-2]
        return string

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
        input_layer: Input[T] = self.layers[0]  # pyright: ignore[reportAssignmentType]
        x, y_targ = input_layer._sync_x_y(x_batch, y_batch)
        has_batch = self.real_batch_size > 0

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

        metrics = None
        metrics, _ = self._compute_metrics_funcs(x, y_targ, loss, self.real_batch_size)
        assert metrics is not None

        return metrics

    def _update_status(
        self,
        pbar: tqdm | None,
        batch_loss: np.ndarray,
        global_loss: np.ndarray,
        output_prefix: str,
        current_round: int,
        delta: float = -1,
        prev_string: str = "",
    ) -> str:
        """
        Updates the progress bar and internal performance counters.

        Args:
            pbar: Tqdm progress bar instance.
            local_loss: Loss/metrics for the current batch.
            global_loss: Accumulated loss/metrics.
            output_prefix: Prefix for logging.
            current_round: The current training/evaluation round.
            delta: Time taken for the batch.
            prev_string: Previous status string.

        Returns:
            Updated total metrics, updated count, and formatted status string.
        """

        part = Dataset.Part[output_prefix.strip("_").upper()]

        string = self._format_metrics(
            metric=global_loss,
            prefix=output_prefix,
        )

        self.perf_counter._add_time_and_batch_size(part, current_round, delta, batch_loss[-1])

        if self.comm_rank == 0:
            # NOTE: pbar is a 'tqdm', it only is None in self.comm_rank != 0
            assert pbar
            pbar.set_postfix_str(s=f"{prev_string}{string}", refresh=True)
            # if part != Dataset.Part.VAL:
            pbar.update(int(global_loss[-1]) - pbar.n)

        return string

    def _evalutate_round(
        self,
        pbar: tqdm | None,
        batch_generator: Generator[tuple[np.ndarray, np.ndarray]],
        model_sync_count: int,
        batches_min: float,
        local_loss: np.ndarray,
        global_loss: np.ndarray,
        terminate: bool = False,
        prev_string: str = "",
        out_prefix: str = "",
    ) -> tuple[int, bool, str]:
        """
        Executes a single evaluation round over the provided batch generator.

        Returns:
            A tuple containing updated total loss, sync count, sync status, and status string.
        """
        sync_epoch = False
        string = ""
        part = Dataset.Part[out_prefix.rstrip("_").upper()]

        for i_batch, (x_batch, y_batch) in enumerate(batch_generator):
            if terminate:
                x_batch = x_batch[:0]
                y_batch = y_batch[:0]
            local_batch_size = x_batch.shape[0]

            sync_model = (self.model_sync_freq <= 0) or (
                model_sync_count % self.model_sync_freq == 0
            )

            if sync_model:
                sync_epoch = True

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
            batch_loss = self._evaluate_batch(x_batch, y_batch, sync_model=sync_model)
            toc = timer()
            delta = toc - tic

            local_loss += batch_loss

            if sync_model:
                if self.comm:
                    local_loss = self.comm.allreduce(local_loss)
                global_loss += local_loss
                local_loss.fill(0)

            string = self._update_status(
                pbar=pbar,
                batch_loss=batch_loss,
                global_loss=global_loss,
                output_prefix=out_prefix,
                current_round=self._evaluate_round,
                delta=delta,
                prev_string=prev_string,
            )

        # Increment self._evaluate_round
        self._evaluate_round += 1
        return (model_sync_count, sync_epoch, string)

    def evaluate(self) -> None:
        """
        Runs the full evaluation process on the test dataset.

        Args:
            bar_width: Width of the progress bar.
        """
        self._ensure_model_runnable()

        if self.use_cudnn and self.y_batch is None:
            assert gpuarray and self.cudnn_dtype
            tensor_ary = TensorArray(
                gpuarray.empty((self.batch_size, *self.layers[-1].shape), self.dtype),
                self.tensor_format,
                self.cudnn_dtype,
            )
            self.y_batch = tensor_ary  # pyright: ignore[reportAttributeAccessIssue]

        self.comm_nsamples = tuple(
            zip(
                *(
                    self.comm.allgather(self.dataset._local_nsamples)
                    if self.comm
                    else [self.dataset._local_nsamples]
                )
            )
        )

        test_batch_generator = self.dataset.get_test_generator()
        test_batches_min: float = min(self.comm_nsamples[Dataset.Part.TEST]) / (
            self.batch_size * self.nprocs
        )

        test_local_loss = np.zeros(len(self.metrics_funcs) + 2, np.object_)
        test_global_loss = np.zeros(len(self.metrics_funcs) + 2, np.object_)

        if self.comm_rank == 0:
            pbar = tqdm(
                file=TqdmLogger(),
                total=sum(self.comm_nsamples[Dataset.Part.TEST]),
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
            local_loss=test_local_loss,
            global_loss=test_global_loss,
            out_prefix=f"{Dataset.Part.TEST._name_.lower()}_",
        )

        if self.comm_rank == 0:
            assert pbar
            pbar.close()
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

        if self.use_blocking_mpi:
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
