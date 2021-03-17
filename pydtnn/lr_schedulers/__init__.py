"""
PyDTNN LR Schedulers

If you want to add a new LR Scheduler:
    1) create a new Python file in this directory,
    2) define your LR Scheduler class as derived from LRScheduler,
    3) and, optionally, import your LR Scheduler on this file.

"""
#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021 Universitat Jaume I
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

from .lr_scheduler import LRScheduler
from .early_stopping import EarlyStopping
from .model_checkpoint import ModelCheckpoint
from .reduce_lr_every_n_epochs import ReduceLREveryNEpochs
from .reduce_lr_on_plateau import ReduceLROnPlateau
from .stop_at_loss import StopAtLoss
from .warm_up_lr_scheduler import WarmUpLRScheduler
from ..utils import get_derived_classes

# Aliases
early_stopping = EarlyStopping
model_checkpoint = ModelCheckpoint
reduce_lr_every_nepochs = ReduceLREveryNEpochs
reduce_lr_on_plateau = ReduceLROnPlateau
stop_at_loss = StopAtLoss
warm_up = WarmUpLRScheduler

# Search this module for LRScheduler derived classes and expose them
get_derived_classes(LRScheduler, locals())


def get_lr_schedulers(model):
    """Get LR Scheduler objects from model attributes"""
    lr_schedulers = []
    for lr_sched in model.lr_schedulers_names.split(","):
        if lr_sched == "warm_up":
            lrs = WarmUpLRScheduler(model.warm_up_epochs,
                                    model.learning_rate / model.mpi_processes,
                                    model.learning_rate)
        elif lr_sched == "early_stopping":
            lrs = EarlyStopping(model.early_stopping_metric,
                                model.early_stopping_patience)
        elif lr_sched == "reduce_lr_on_plateau":
            lrs = ReduceLROnPlateau(model.reduce_lr_on_plateau_metric,
                                    model.reduce_lr_on_plateau_factor,
                                    model.reduce_lr_on_plateau_patience,
                                    model.reduce_lr_on_plateau_min_lr)
        elif lr_sched == "reduce_lr_every_nepochs":
            lrs = ReduceLREveryNEpochs(model.reduce_lr_every_nepochs_factor,
                                       model.reduce_lr_every_nepochs_nepochs,
                                       model.reduce_lr_every_nepochs_min_lr)
        elif lr_sched == "stop_at_loss":
            lrs = StopAtLoss(model.stop_at_loss_metric,
                             model.stop_at_loss_threshold)
        elif lr_sched == "model_checkpoint":
            lrs = ModelCheckpoint(model.model_checkpoint_metric,
                                  model.model_checkpoint_save_freq)
        else:
            raise ValueError(f"LRScheduler '{model.optimizer}' not recognized.")
        lr_schedulers.append(lrs)
    return lr_schedulers
