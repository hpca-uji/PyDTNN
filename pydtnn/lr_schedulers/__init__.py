"""
PyDTNN LR Schedulers

If you want to add a new LR Scheduler:
    1) create a new Python file in this directory,
    2) define your LR Scheduler class as derived from LRScheduler,
    3) and, optionally, import your LR Scheduler on this file.

"""
from .lr_scheduler import LRScheduler
# NOTE: The following import is necessary for other imports:
from .lr_scheduler_with_loss_or_metric import LRSchedulerWithLossOrMetric
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


def get_lr_schedulers(model) -> list[LRScheduler]:
    """Get LR Scheduler objects from model attributes"""
    lr_schedulers = []
    # NOTE: All this parameters came from Parser
    for lr_sched in filter(None, model.lr_schedulers_names.split(",")):
        match lr_sched:
            case "warm_up":
                lrs = WarmUpLRScheduler(model,
                                        model.warm_up_epochs,
                                        model.learning_rate / model.nprocs,
                                        model.learning_rate)
            case "early_stopping":
                lrs = EarlyStopping(model,
                                    model.early_stopping_metric,
                                    model.early_stopping_patience,
                                    model.early_stopping_minimize)
            case "reduce_lr_on_plateau":
                lrs = ReduceLROnPlateau(model,
                                        model.reduce_lr_on_plateau_metric,
                                        model.reduce_lr_on_plateau_factor,
                                        model.reduce_lr_on_plateau_patience,
                                        model.reduce_lr_on_plateau_min_lr)
            case "reduce_lr_every_nepochs":
                lrs = ReduceLREveryNEpochs(model,
                                           model.reduce_lr_every_nepochs_factor,
                                           model.reduce_lr_every_nepochs_nepochs,
                                           model.reduce_lr_every_nepochs_min_lr)
            case "stop_at_loss":
                lrs = StopAtLoss(model,
                                 model.stop_at_loss_metric,
                                 model.stop_at_loss_threshold)
            case "model_checkpoint":
                lrs = ModelCheckpoint(model,
                                      model.model_checkpoint_metric,
                                      model.model_checkpoint_save_freq)
            case _:
                raise SystemExit(f"LRScheduler '{model.optimizer}' not supported yet!")
        lr_schedulers.append(lrs)
    return lr_schedulers
