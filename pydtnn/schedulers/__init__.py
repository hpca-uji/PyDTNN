"""
PyDTNN LR Schedulers

If you want to add a new LR Scheduler:
    1) create a new Python file in this directory,
    2) define your LR Scheduler class as derived from LRScheduler,
    3) and, optionally, import your LR Scheduler on this file.

"""
from pydtnn.schedulers.lr_scheduler import LRScheduler as _LRScheduler
# NOTE: The following import is necessary for other imports:
from pydtnn.schedulers.early_stopping import EarlyStopping as _EarlyStopping
from pydtnn.schedulers.model_checkpoint import ModelCheckpoint as _ModelCheckpoint
from pydtnn.schedulers.reduce_lr_every_n_epochs import ReduceLREveryNEpochs as _ReduceLREveryNEpochs
from pydtnn.schedulers.reduce_lr_on_plateau import ReduceLROnPlateau as _ReduceLROnPlateau
from pydtnn.schedulers.stop_at_loss import StopAtLoss as _StopAtLoss
from pydtnn.schedulers.warm_up_lr_scheduler import WarmUpLRScheduler as _WarmUpLRScheduler


def get_schedulers(model) -> list[_LRScheduler]:
    """Get Scheduler objects from model attributes"""
    schedulers = []
    # NOTE: All this parameters came from Parser
    for scheduler_name in filter(None, model.schedulers_names.split(",")):
        match scheduler_name:
            case "warm_up":
                scheduler = _WarmUpLRScheduler(model.warm_up_epochs,
                                               model.learning_rate / model.nprocs,
                                               model.learning_rate)
            case "early_stopping":
                scheduler = _EarlyStopping(model.early_stopping_metric,
                                           model.early_stopping_patience,
                                           model.early_stopping_minimize)
            case "reduce_lr_on_plateau":
                scheduler = _ReduceLROnPlateau(model.reduce_lr_on_plateau_metric,
                                               model.reduce_lr_on_plateau_factor,
                                               model.reduce_lr_on_plateau_patience,
                                               model.reduce_lr_on_plateau_min_lr)
            case "reduce_lr_every_nepochs":
                scheduler = _ReduceLREveryNEpochs(model.reduce_lr_every_nepochs_factor,
                                                  model.reduce_lr_every_nepochs_nepochs,
                                                  model.reduce_lr_every_nepochs_min_lr)
            case "stop_at_loss":
                scheduler = _StopAtLoss(model.stop_at_loss_metric,
                                        model.stop_at_loss_threshold)
            case "model_checkpoint":
                scheduler = _ModelCheckpoint(model.model_checkpoint_metric,
                                             model.model_checkpoint_save_freq)
            case _:
                raise SystemExit(f"LRScheduler '{model.optimizer}' not supported yet!")
        schedulers.append(scheduler)
    return schedulers
