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

    def __init__(self, warmup_epochs=5, base_lr=1e-4, init_lr=1e-3, verbose=True):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.init_lr = init_lr
        self.verbose = verbose
        self.batch_count = 0

    def on_batch_begin(self, model, optimizer, rank):
        warmup_batches = int(model.steps_per_epoch) * self.warmup_epochs
        if self.batch_count <= warmup_batches:
            optimizer.learning_rate = self.base_lr + self.batch_count * ((self.init_lr - self.base_lr) / warmup_batches)
            self.batch_count += 1
            # if self.verbose and rank == 0:
            #     print("LRScheduler %s: setting learning rate to %.8f" % \
            #         (type(self).__name__, optimizer.learning_rate))
