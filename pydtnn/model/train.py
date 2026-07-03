"""Training code for the PyDTNN model"""

import enum
import logging
import time
from collections.abc import Generator
from timeit import default_timer as timer
from typing import Any

import numpy as np
from tqdm import tqdm

from pydtnn import MPI, gpuarray
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.datasets.abstract import Dataset
from pydtnn.layers.input import Input
from pydtnn.model.base import Base
from pydtnn.model.eval import Eval
from pydtnn.schedulers import select as select_scheduler
from pydtnn.schedulers.abstract.scheduler import Scheduler
from pydtnn.tracers.events import (PYDTNN_EVENT_FINISHED, PYDTNN_MDL_EVENT,
                                   PYDTNN_MDL_EVENTS, MdlEventEnum)
from pydtnn.utils.logs import TqdmLogger
from pydtnn.utils.constants import Array

__all__ = ("Train",)

logger = logging.getLogger(__name__)


class Train[T: Array](Eval[T]):  # noqa: D101 (generics not detected)
    """
    Handles the training process for a model, including weight synchronization,
    gradient updates, and training loop management.
    """

    class SyncParticipation(enum.StrEnum):
        """Defines strategies for node participation in model synchronization."""

        ALL = enum.auto()
        AVAIL2ALL = enum.auto()

    class SyncAlgorithm(enum.StrEnum):
        """Defines algorithms for weight aggregation during synchronization."""

        AVG = enum.auto()
        WAVG = enum.auto()
        INVAVG = enum.auto()

    def __init__(self, **kwargs: Any) -> None:
        """Initializes the training instance with synchronization parameters and schedulers."""
        super().__init__(**kwargs)
        self._training_round: int = 0
        # Synchronization parameters
        # NOTE: This parameter come from Parser.
        self.model_sync_algo = self.SyncAlgorithm(self.model_sync_algo)

        # NOTE: This parameter come from Parser.
        self.model_sync_participation = self.SyncParticipation(
            self.kwargs["model_sync_participation"]
        )

        self.schedulers: list[Scheduler] = [
            select_scheduler(scheduler_name).from_model(self)
            for scheduler_name in filter(None, self.schedulers_names.split(","))
        ]
        for scheduler in self.schedulers:
            scheduler.model = self  # type: ignore (It's fine)

    def _train_batch(
        self, x_batch: np.ndarray, y_batch: np.ndarray, sync_model: bool = True
    ) -> np.ndarray:
        """Executes a single training batch including forward pass, backward pass, and weight updates."""
        self.mode = Base.Mode.TRAIN

        # Schedulers begin
        for sched in self.schedulers:
            sched.on_batch_begin()

        self.real_batch_size = x_batch.shape[0]
        input_layer: Input[T] = self.layers[0]  # type: ignore (casting to the right type)
        x, y_targ = input_layer._sync_x_y(x_batch, y_batch)

        has_batch = x_batch.shape[0] > 0

        if has_batch:
            # Forward pass (FP)
            for layer in self.layers:
                self.tracer.emit_event(
                    PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + MdlEventEnum.FORWARD
                )
                x = layer.forward(x)
                self.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)
            loss, dx = self.loss_func.compute(x, y_targ, self.real_batch_size)
        else:
            if y_targ.shape[0] != x_batch.shape[0]:
                raise ValueError(
                    f"y_targ.shape[0] ({y_targ.shape[0]})"
                    " and x_batch.shape[0] ({x_batch.shape[0]})"
                    " must have the same value."
                )
            loss, dx = 0.0, y_targ

        total_metrics = None
        total_metrics, _ = self._compute_metrics_funcs(x, y_targ, loss, comm=sync_model)
        assert total_metrics is not None
        self.total_metrics = total_metrics

        if has_batch:
            # Backward pass (BP)
            for layer in reversed(self.layers):
                self.tracer.emit_event(
                    PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + MdlEventEnum.BACKWARD
                )
                dx = layer.backward(dx)
                self.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)

        if self.stream:
            self.stream.synchronize()  # type: ignore

        # Gradient update (GU)
        if self.model_sync_freq >= 0 and sync_model:
            self._weight_update(
                gradient=True, blocking=self.blocking_mpi, pipeline=self.parallel_pipeline
            )

        if has_batch or sync_model:
            # Optimizer
            for layer in self.layers:
                self.tracer.emit_event(
                    PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + MdlEventEnum.UPDATE_DW
                )
                layer.update_weights(self.optimizer)
                self.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)

        # Weight update (WU)
        if self.model_sync_freq > 0 and sync_model:
            self._weight_update(
                gradient=False, blocking=self.blocking_mpi, pipeline=self.parallel_pipeline
            )

        if self.enable_cudnn:
            for layer in self.layers:
                if layer.grad_vars and layer.stream_2:
                    layer.stream_2.synchronize()  # type: ignore

        # Schedulers end
        for sched in self.schedulers:
            sched.on_batch_end(self)

        return self.total_metrics

    def _train_round(
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
        """Executes a full training round over the provided batch generator."""
        sync_epoch = False
        string = ""

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

            if i_batch >= batches_min and sync_model:
                rank_mask = (
                    self.comm.allgather(min(1, local_batch_size))
                    if self.comm
                    else [min(1, local_batch_size)]
                )
            else:
                rank_mask = [1] * self.comm_size
            rank_avail = sum(rank_mask)

            if rank_avail <= 0:
                break

            if rank_avail < self.model_sync_min_avail:
                sync_model = False

            self.rank_weight = self._compute_rank_weight(rank_mask, Dataset.Part.TRAIN)

            tic = timer()
            train_batch_loss = self._train_batch(x_batch, y_batch, sync_model=sync_model)
            toc = timer()
            delta = toc - tic

            if local_batch_size <= 0:
                if self.comm_rank == 0:
                    # (in comm_rank 0 , pbar is not None)
                    pbar.set_postfix_str(s=f"{string}, waiting…", refresh=True)  # type: ignore
                continue

            total_loss, total_size, string = self._update_status(
                pbar=pbar,
                batch_loss=train_batch_loss,
                total_loss=total_loss,
                total_size=total_size,
                batch_size=batch_size,
                output_prefix=out_prefix,
                current_round=self._training_round,
                delta=delta,
                prev_string=prev_string,
            )

        # Increment self._train_round
        self._training_round += 1
        return (total_loss, total_size, model_sync_count, sync_epoch, string)

    def train(self) -> dict[str, list[np.ndarray]]:
        """Runs the full training process over multiple epochs."""
        self._ensure_model_runnable()

        # If working with CUDA, self.y_batch must be in a GPU's data structure.
        if self.enable_cudnn and self.y_batch is None:
            assert gpuarray and self.cudnn_dtype
            tensor_ary = TensorArray(
                gpuarray.empty((self.batch_size, *self.layers[-1].shape), self.dtype),
                self.tensor_format,
                self.cudnn_dtype,
            )
            self.y_batch = tensor_ary  # type: ignore

        self.history = {
            lm: []
            for lm in [f"{Dataset.Part.TRAIN._name_.lower()}_{m}" for m in self.loss_and_metrics]
            + [f"{Dataset.Part.VAL._name_.lower()}_{m}" for m in self.loss_and_metrics]
        }

        self.comm_nsamples = list(
            zip(
                *(
                    self.comm.allgather(self.dataset._nsamples)
                    if self.comm
                    else [self.dataset._nsamples]
                )
            )
        )

        terminate = False  # True: ends the following loop.
        global_terminate = False

        model_sync_count = 0
        train_batches_min = min(self.comm_nsamples[Dataset.Part.TRAIN]) / (
            self.batch_size * self.nprocs
        )
        val_batches_min = min(self.comm_nsamples[Dataset.Part.VAL]) / (
            self.batch_size * self.nprocs
        )

        for epoch in range(self.num_epochs):
            train_batch_generator, val_batch_generator = self.dataset.get_train_val_generator()
            sync_epoch = False

            train_total_loss, train_total_size = np.zeros(len(self.loss_and_metrics)), 0
            val_total_loss, val_total_size = np.zeros(len(self.loss_and_metrics)), 0

            if self.comm_rank == 0:
                string = ""
                fmt = "%%%dd" % (len(str(self.num_epochs)))
                epoch_string = "Epoch %s/%s" % (fmt, fmt)
                pbar = tqdm(
                    file=TqdmLogger(),
                    total=self.dataset.train_nsamples,
                    ascii=" ▁▂▃▄▅▆▇█",
                    smoothing=0.3,
                    desc=epoch_string % (epoch + 1, self.num_epochs),
                    unit=" samples",
                )
            else:
                pbar = None

            for sched in self.schedulers:
                sched.on_epoch_begin(self, self.rank)

            # --- TRAIN ---
            train_total_loss, train_total_size, model_sync_count, train_sync_epoch, string = (
                self._train_round(
                    pbar=pbar,
                    batch_generator=train_batch_generator,
                    model_sync_count=model_sync_count,
                    batches_min=train_batches_min,
                    total_loss=train_total_loss,
                    total_size=train_total_size,
                    prev_string="",
                    out_prefix=f"{Dataset.Part.TRAIN._name_.lower()}_",
                )
            )
            sync_epoch = sync_epoch or train_sync_epoch
            train_string = string

            for c in range(len(self.loss_and_metrics)):
                self.history[
                    f"{Dataset.Part.TRAIN._name_.lower()}_" + self.loss_and_metrics[c]
                ].append(train_total_loss[c])

            # --- VAL ---
            val_total_loss, val_total_size, model_sync_count, val_sync_epoch, string = (
                self._evalutate_round(
                    pbar=pbar,
                    batch_generator=val_batch_generator,
                    model_sync_count=model_sync_count,
                    batches_min=val_batches_min,
                    total_loss=val_total_loss,
                    total_size=val_total_size,
                    prev_string=f"{train_string}, ",
                    out_prefix=f"{Dataset.Part.VAL._name_.lower()}_",
                )
            )
            sync_epoch = sync_epoch or val_sync_epoch

            # if self.comm_rank == 0:  # All nodes must have history, not only the 0.
            for c in range(len(self.loss_and_metrics)):
                self.history[
                    f"{Dataset.Part.VAL._name_.lower()}_" + self.loss_and_metrics[c]
                ].append(val_total_loss[c])

            for sched in self.schedulers:
                sched.on_epoch_end(train_total_loss, val_total_loss)
                if sched.stop_training:
                    terminate = True

            if self.comm_rank == 0:
                pbar.close()  # type: ignore (Here is a 'tqdm', only is None in self.comm_rank != 0)
                # Sleep for half a second to allow pbar to write its output before returning
                time.sleep(0.5)

            for c in range(len(self.loss_and_metrics)):
                if not self.loss_and_metrics_format[c]:
                    logger.info(
                        f"{Dataset.Part.TRAIN._name_.lower()}_{self.loss_and_metrics[c]}:"
                        f" {train_total_loss[c]}"
                    )
            for c in range(len(self.loss_and_metrics)):
                if not self.loss_and_metrics_format[c]:
                    logger.info(
                        f"{Dataset.Part.VAL._name_.lower()}_{self.loss_and_metrics[c]}:"
                        f" {val_total_loss[c]}"
                    )

            if sync_epoch:
                if self.comm is not None:
                    op = MPI.LAND  # type: ignore
                    global_terminate = self.comm.allreduce(terminate, op=op)
                else:
                    global_terminate = terminate

            if global_terminate:
                break

        # End pipelines
        self._model_reduce_wait(gradient=True)
        self._model_reduce_wait(gradient=False)

        # Synchronize model
        if self.final_model_sync:
            self._weight_update(gradient=False, blocking=self.blocking_mpi)

        self.tracer.define_event_types(self)
        return self.history
