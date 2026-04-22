import time

import numpy as np
from timeit import default_timer as timer

from tqdm import tqdm
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.datasets.dataset import Dataset
from pydtnn.tracers.events import PYDTNN_EVENT_FINISHED, PYDTNN_MDL_EVENT, PYDTNN_MDL_EVENTS, PYDTNN_MDL_EVENT_enum
from pydtnn.utils.constants import Array
from pydtnn import gpuarray

from pydtnn._model.model_init import Model_Init as Model

import logging
logger = logging.getLogger(__name__)

class Model_Eval[T: Array](Model[T]):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Private attributes
        self._evaluate_round: int = 0
    
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
        self.mode = Model.Mode.EVALUATE

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

        total_metrics, _ = self._compute_metrics_funcs(y_pred, y_targ, loss, comm=sync_model)
        assert total_metrics is not None
        self.total_metrics = total_metrics

        return self.total_metrics

    def evaluate(self, bar_width=Model.BAR_WIDTH):
        self._ensure_model_runable()

        if self.enable_cudnn and self.y_batch is None:
            assert gpuarray and self.cudnn_dtype
            tensor_ary = TensorArray(
                gpuarray.empty((self.batch_size, *self.layers[-1].shape), self.dtype),
                self.tensor_format, self.cudnn_dtype)
            self.y_batch = tensor_ary  # type: ignore

        self.comm_nsamples = list(zip(*self.comm.allgather(self.dataset._nsamples) if self.comm else [self.dataset._nsamples]))

        test_batches_min = min(self.comm_nsamples[Dataset.Part.TEST]) / (self.batch_size * self.nprocs)

        test_batch_generator = self.dataset.get_test_generator()

        if self.comm_rank == 0:
            test_total_loss, test_batch_count = np.zeros(len(self.loss_and_metrics)), 0
            pbar = tqdm(total=self.dataset.test_nsamples, ncols=bar_width,
                        ascii=" ▁▂▃▄▅▆▇█", smoothing=0.3,
                        desc="Testing", unit=" samples")

        model_sync_count = 0
        for i_batch, (x_batch, y_batch, batch_size) in enumerate(test_batch_generator):
            local_batch_size = x_batch.shape[0]

            sync_model = (self.model_sync_freq <= 0) or (model_sync_count % self.model_sync_freq == 0)

            if model_sync_count == 0 and not self.initial_model_sync:
                sync_model = False

            model_sync_count += 1

            if i_batch < test_batches_min:
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

            if batch_size <= 0:
                continue

            if self.comm_rank == 0:
                # noinspection PyUnboundLocalVariable
                test_total_loss, test_batch_count, string = \
                    self._update_running_average(test_batch_loss, test_total_loss, test_batch_count, batch_size, prefix="test_")
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
