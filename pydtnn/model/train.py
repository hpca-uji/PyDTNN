"""Training code for the PyDTNN model"""

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
from pydtnn.utils.constants import Array
from pydtnn.utils.logs import TqdmLogger

__all__ = ("Train",)

logger = logging.getLogger(__name__)


class Train[T: Array](Eval[T]):  # noqa: D101 (generics not detected)
    """
    Handles the training process for a model, including weight synchronization,
    gradient updates, and training loop management.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initializes the training instance with synchronization parameters and schedulers."""
        super().__init__(**kwargs)
        self._training_round: int = 0
        # Synchronization parameters
        # NOTE: This parameter come from Parser.
        self.model_sync_algo = Base.SyncAlgorithm(self.model_sync_algo)

        # NOTE: This parameter come from Parser.
        self.model_sync_participation = Base.SyncParticipation(self.model_sync_participation)

        self.schedulers: list[Scheduler] = [
            select_scheduler(scheduler_name).from_model(self)
            for scheduler_name in self.schedulers_names
        ]
        for scheduler in self.schedulers:
            scheduler.model = self  # pyright: ignore[reportAttributeAccessIssue]

    def _train_batch(  # noqa: C901
        self, x_batch: np.ndarray, y_batch: np.ndarray, sync_model: bool = True
    ) -> np.ndarray:
        """Executes a single training batch including forward pass, backward pass, and weight updates."""
        self.mode = Base.Mode.TRAIN

        # Schedulers begin
        for sched in self.schedulers:
            sched.on_batch_begin()

        self.real_batch_size = x_batch.shape[0]
        input_layer: Input[T] = self.layers[0]  # pyright: ignore[reportAssignmentType]
        x, y_targ = input_layer._sync_x_y(x_batch, y_batch)

        has_batch = x_batch.shape[0] > 0

        dx: T
        if has_batch:
            # Forward pass (FP)
            for layer in self.layers:
                self.tracer.emit_event(
                    PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + MdlEventEnum.FORWARD
                )
                x = layer.forward(x)
                self.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)
            loss, dx = self.loss_func.compute(x, y_targ)
        else:
            if y_targ.shape[0] != x_batch.shape[0]:
                raise ValueError(
                    f"y_targ.shape[0] ({y_targ.shape[0]})"
                    " and x_batch.shape[0] ({x_batch.shape[0]})"
                    " must have the same value."
                )
            loss, dx = 0.0, y_targ

        metrics = None
        metrics, _ = self._compute_metrics_funcs(x, y_targ, loss, self.real_batch_size)
        assert metrics is not None

        if has_batch:
            # Backward pass (BP)
            for layer in reversed(self.layers):
                self.tracer.emit_event(
                    PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + MdlEventEnum.BACKWARD
                )
                dx = layer.backward(dx)
                self.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)

        if self.stream:
            self.stream.synchronize()

        # Gradient update (GU)
        if self.model_sync_freq >= 0 and sync_model:
            self._weight_update(
                gradient=True, blocking=self.use_blocking_mpi, pipeline=self.parallel_pipeline
            )

        # Optimizer
        for layer in self.layers:
            self.tracer.emit_event(
                PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + MdlEventEnum.UPDATE_DW
            )
            layer.update_weights(self.optimizer, has_batch, sync_model)
            self.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)

        # Weight update (WU)
        if self.model_sync_freq > 0 and sync_model:
            self._weight_update(
                gradient=False, blocking=self.use_blocking_mpi, pipeline=self.parallel_pipeline
            )

        if self.use_cudnn:
            for layer in self.layers:
                if layer.grad_vars and layer.stream_2:
                    layer.stream_2.synchronize()

        # Schedulers end
        for sched in self.schedulers:
            sched.on_batch_end(self)

        return metrics

    def _train_round(
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
    ) -> tuple[int, bool, str]:  # noqa: C901
        """Executes a full training round over the provided batch generator."""
        sync_epoch = False
        string = ""

        for i_batch, (x_batch, y_batch) in enumerate(batch_generator):
            if terminate:
                x_batch = x_batch[:0]
                y_batch = y_batch[:0]

            local_batch_size = x_batch.shape[0]
            sync_model = (self.model_sync_freq <= 0) \
                or (model_sync_count % self.model_sync_freq == 0)

            if sync_model:
                sync_epoch = True

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
            batch_loss = self._train_batch(x_batch, y_batch, sync_model=sync_model)
            toc = timer()
            delta = toc - tic

            local_loss += batch_loss

            # if local_batch_size <= 0:
            #     if self.comm_rank == 0:
            #         assert pbar
            #         pbar.set_postfix_str(s=f"{string}, waiting…", refresh=True)

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
                current_round=self._training_round,
                delta=delta,
                prev_string=prev_string,
            )

        # Increment self._train_round
        self._training_round += 1
        return (model_sync_count, sync_epoch, string)

    def train(self) -> dict[str, list[np.ndarray]]:  # noqa: C901
        """Runs the full training process over multiple epochs."""
        self._ensure_runnable()

        # If working with CUDA, self.y_batch must be in a GPU's data structure.
        if self.use_cudnn and self.y_batch is None:
            assert gpuarray and self.cudnn_dtype
            tensor_ary = TensorArray(
                gpuarray.empty((self.batch_size, *self.layers[-1].shape), self.dtype),
                self.tensor_format,
                self.cudnn_dtype,
            )
            self.y_batch = tensor_ary  # pyright: ignore[reportAttributeAccessIssue]

        self.history = {
            lm: []
            for lm in [f"{Dataset.Part.TRAIN._name_.lower()}_{m}" for m in self.loss_and_metrics]
            + [f"{Dataset.Part.VAL._name_.lower()}_{m}" for m in self.loss_and_metrics]
        }

        self.comm_nsamples = tuple(
            zip(
                *(
                    self.comm.allgather(self.dataset._local_nsamples)
                    if self.comm
                    else [self.dataset._local_nsamples]
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

        # Synchronize model
        if self.initial_model_sync:
            self._weight_update(gradient=True, blocking=self.use_blocking_mpi)
            self._weight_update(gradient=False, blocking=self.use_blocking_mpi)

        for epoch in range(self.num_epochs):
            train_batch_generator, val_batch_generator = self.dataset.get_train_val_generator()
            sync_epoch = False

            train_local_loss = np.zeros(len(self.metrics_funcs) + 2, dtype=np.object_)
            val_local_loss = np.zeros(len(self.metrics_funcs) + 2, dtype=np.object_)

            train_global_loss = np.zeros(len(self.metrics_funcs) + 2, dtype=np.object_)
            val_global_loss = np.zeros(len(self.metrics_funcs) + 2, dtype=np.object_)

            for sched in self.schedulers:
                sched.on_epoch_begin()

            # --- TRAIN ---
            if self.comm_rank == 0:
                fmt = "%%0%dd" % (len(str(self.num_epochs)))
                epoch_string = "Training   (%s/%s)" % (fmt, fmt)
                pbar = tqdm(
                    file=TqdmLogger(),
                    total=sum(self.comm_nsamples[Dataset.Part.TRAIN]),
                    ascii=" ▁▂▃▄▅▆▇█",
                    smoothing=0.3,
                    desc=epoch_string % (epoch + 1, self.num_epochs),
                    unit=" samples",
                )
            else:
                pbar = None

            model_sync_count, train_sync_epoch, string = (
                self._train_round(
                    pbar=pbar,
                    batch_generator=train_batch_generator,
                    model_sync_count=model_sync_count,
                    batches_min=train_batches_min,
                    local_loss=train_local_loss,
                    global_loss=train_global_loss,
                    prev_string="",
                    out_prefix=f"{Dataset.Part.TRAIN._name_.lower()}_",
                )
            )
            train_global_loss[:-1] /= train_global_loss[-1]
            sync_epoch = sync_epoch or train_sync_epoch

            for c in range(len(self.loss_and_metrics)):
                self.history[
                    f"{Dataset.Part.TRAIN._name_.lower()}_" + self.loss_and_metrics[c]
                ].append(train_global_loss[c])

            if self.comm_rank == 0:
                assert pbar
                pbar.close()
                # Sleep for half a second to allow pbar to write its output before returning
                time.sleep(0.5)

            # --- VAL ---
            if self.comm_rank == 0:
                fmt = "%%0%dd" % (len(str(self.num_epochs)))
                epoch_string = "Validating (%s/%s)" % (fmt, fmt)
                pbar = tqdm(
                    file=TqdmLogger(),
                    total=sum(self.comm_nsamples[Dataset.Part.VAL]),
                    ascii=" ▁▂▃▄▅▆▇█",
                    smoothing=0.3,
                    desc=epoch_string % (epoch + 1, self.num_epochs),
                    unit=" samples",
                )
            else:
                pbar = None

            model_sync_count, val_sync_epoch, string = (
                self._evalutate_round(
                    pbar=pbar,
                    batch_generator=val_batch_generator,
                    model_sync_count=model_sync_count,
                    batches_min=val_batches_min,
                    local_loss=val_local_loss,
                    global_loss=val_global_loss,
                    prev_string="",
                    out_prefix=f"{Dataset.Part.VAL._name_.lower()}_",
                )
            )
            val_global_loss[:-1] /= val_global_loss[-1]
            sync_epoch = sync_epoch or val_sync_epoch

            for c in range(len(self.loss_and_metrics)):
                self.history[
                    f"{Dataset.Part.VAL._name_.lower()}_" + self.loss_and_metrics[c]
                ].append(val_global_loss[c])

            if self.comm_rank == 0:
                assert pbar
                pbar.close()
                # Sleep for half a second to allow pbar to write its output before returning
                time.sleep(0.5)

            for sched in self.schedulers:
                sched.on_epoch_end(train_global_loss, val_global_loss)
                if sched.stop_training:
                    terminate = True

            for c in range(len(self.loss_and_metrics)):
                if not self.loss_and_metrics_format[c]:
                    logger.info(
                        f"{Dataset.Part.TRAIN._name_.lower()}_{self.loss_and_metrics[c]}:"
                        f" {train_global_loss[c]}"
                    )
            for c in range(len(self.loss_and_metrics)):
                if not self.loss_and_metrics_format[c]:
                    logger.info(
                        f"{Dataset.Part.VAL._name_.lower()}_{self.loss_and_metrics[c]}:"
                        f" {val_global_loss[c]}"
                    )

            if sync_epoch:
                if self.comm:
                    op = MPI.LAND
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
            self._weight_update(gradient=True, blocking=self.use_blocking_mpi)
            self._weight_update(gradient=False, blocking=self.use_blocking_mpi)

        self.tracer.define_event_types(self)
        return self.history
