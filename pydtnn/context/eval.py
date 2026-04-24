import time
from typing import Any, Generator

import numpy as np
from timeit import default_timer as timer

from pydtnn import MPI
from tqdm import tqdm
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.datasets.dataset import Dataset
from pydtnn.tracers.events import PYDTNN_EVENT_FINISHED, PYDTNN_MDL_EVENT, PYDTNN_MDL_EVENTS, PYDTNN_MDL_EVENT_enum
from pydtnn.utils.constants import Array
from pydtnn.context.utils import BAR_WIDTH
from pydtnn import gpuarray

from pydtnn.context.sync import Sync

import logging
logger = logging.getLogger(__name__)


class Eval[T: Array](Sync[T]):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Private attributes
        self._evaluate_round: int = 0

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

    def _evaluate_batch(self, x_batch: np.ndarray, y_batch: np.ndarray, sync_model=True) -> np.ndarray:
        self.mode = Sync.Mode.EVALUATE

        self.real_batch_size = x_batch.shape[0]
        x, y_targ = self.layers[0]._sync_x_y(x_batch, y_batch)

        has_batch = x_batch.shape[0] > 0

        # Forward pass (FP)
        if has_batch:
            for i in range(len(self.layers)):
                self.tracer.emit_event(PYDTNN_MDL_EVENT, self.layers[i].id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.FORWARD)
                x = self.layers[i].forward(x)
                self.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)

            y_pred = self.layers[-1].y
            loss, _ = self.loss_func.compute(y_pred, y_targ, self.real_batch_size)
        else:
            y_pred = self.layers[-1].y
            loss = 0.0
        assert y_pred is not None

        total_metrics = None
        total_metrics, _ = self._compute_metrics_funcs(y_pred, y_targ, loss, comm=sync_model)
        assert total_metrics is not None
        self.total_metrics = total_metrics

        return self.total_metrics

    def _update_status(self, pbar: tqdm, batch_loss: np.ndarray, total_loss: np.ndarray,
                       batch_count: int, batch_size: int, output_prefix: str, delta: float = -1,
                       prev_string: str = "") -> tuple[np.ndarray, int, str]:
        # noinspection PyUnboundLocalVariable
        total_loss, batch_count, string = \
            self._update_running_average(batch_loss, total_loss, batch_count, batch_size, prefix=output_prefix)

        if self.comm_rank == 0:
            # noinspection PyUnboundLocalVariable
            pbar.set_postfix_str(s=f"{prev_string}{string}", refresh=True)
            if delta >= 0:
                pbar.update(batch_size)
                self.perf_counter.add_testing_time_and_batch_size(self._evaluate_round, delta, batch_size)

        return total_loss, batch_count, string
    # ------

    def _evalutate_round(self, pbar: tqdm,
                         batch_generator: Generator[tuple[np.ndarray, np.ndarray, int]],
                         model_sync_count: int,
                         batches_min: float,
                         total_loss: np.ndarray,
                         batch_count: int,
                         terminate: bool = False,
                         prev_string: str = "",
                         out_prefix: str = "") -> tuple[int, bool, str]:
        """
        Return:
            tuple[model_sync_count (int), sync_epoch (bool)]
        """
        sync_epoch = False
        string = ""

        for i_batch, (x_batch, y_batch, batch_size) in enumerate(batch_generator):
            if terminate:
                x_batch = x_batch[:0]
                y_batch = y_batch[:0]
            local_batch_size = x_batch.shape[0]

            sync_model = (self.model_sync_freq <= 0) or (model_sync_count % self.model_sync_freq == 0)

            if sync_model:
                sync_epoch = True

            if model_sync_count == 0 and not self.initial_model_sync:
                sync_model = False

            model_sync_count += 1

            if i_batch < batches_min:
                rank_mask = [1] * self.comm_size
            else:
                rank_mask = self.comm.allgather(min(1, local_batch_size)) if self.comm else [min(1, local_batch_size)]
            rank_avail = sum(rank_mask)

            if rank_avail <= 0:
                break

            if rank_avail < self.model_sync_min_avail:
                sync_model = False

            tic = timer()
            test_batch_loss = self._evaluate_batch(x_batch, y_batch, sync_model=sync_model)
            toc = timer()
            delta = toc - tic

            if out_prefix != "test_":
                delta = -1

            if batch_size <= 0:
                continue

            total_loss, batch_count, string = self._update_status(pbar=pbar, batch_loss=test_batch_loss,
                                                                  total_loss=total_loss, batch_count=batch_count,
                                                                  batch_size=batch_size, output_prefix=out_prefix, delta=delta,
                                                                  prev_string=prev_string)

        return (model_sync_count, sync_epoch, string)
    # -----

    def evaluate(self, bar_width=BAR_WIDTH):
        self._ensure_model_runable()

        if self.enable_cudnn and self.y_batch is None:
            assert gpuarray and self.cudnn_dtype
            tensor_ary = TensorArray(
                gpuarray.empty((self.batch_size, *self.layers[-1].shape), self.dtype),
                self.tensor_format, self.cudnn_dtype)
            self.y_batch = tensor_ary  # type: ignore

        self.comm_nsamples = list(zip(*self.comm.allgather(self.dataset._nsamples) if self.comm else [self.dataset._nsamples]))

        test_batches_min: float = min(self.comm_nsamples[Dataset.Part.TEST]) / (self.batch_size * self.nprocs)

        test_batch_generator = self.dataset.get_test_generator()

        if self.comm_rank == 0:
            test_total_loss, test_batch_count = np.zeros(len(self.loss_and_metrics)), 0
            pbar = tqdm(total=self.dataset.test_nsamples, ncols=bar_width,
                        ascii=" ▁▂▃▄▅▆▇█", smoothing=0.3,
                        desc="Testing", unit=" samples")

        self._evalutate_round(pbar=pbar, batch_generator=test_batch_generator,
                              model_sync_count=0, batches_min=test_batches_min,
                              total_loss=test_total_loss, batch_count=test_batch_count,
                              out_prefix="test_")

        # Increment self._evaluate_round
        self._evaluate_round += 1

        if self.comm_rank == 0:
            pbar.close()
            # Sleep for half a second to allow pbar to write its output before returning
            time.sleep(.5)

        # End pipelines
        self._model_reduce_wait(gradient=True)
        self._model_reduce_wait(gradient=False)
