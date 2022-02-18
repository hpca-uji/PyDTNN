#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
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

from . import LRScheduler


class WarmUpLRScheduler(LRScheduler):
    """
    WarmUpLRScheduler
    """

    def __init__(self, model, warmup_epochs=5, base_lr=1e-4, init_lr=1e-3, verbose=True):
        super().__init__(model, verbose)
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.init_lr = init_lr
        self.batch_count = 0

    def on_batch_begin(self):
        warmup_batches = (self.model.dataset.train_nsamples // (self.model.batch_size * self.model.nprocs)) * self.warmup_epochs
        if self.batch_count <= warmup_batches:
            self.model.optimizer.learning_rate = self.base_lr \
                                                 + self.batch_count * ((self.init_lr - self.base_lr) / warmup_batches)
            self.batch_count += 1
            self.log(f"Setting learning rate to {self.model.optimizer.learning_rate:.8f}.")
