"""
PyDTNN LR Schedulers

If you want to add a new LR Scheduler:
    1) create a new Python file in this directory,
    2) define your LR Scheduler class as derived from LRScheduler,
    3) and, optionally, import your LR Scheduler on this file.

"""
from pydtnn.lr_schedulers.lr_scheduler import LRScheduler as _LRScheduler
# NOTE: The following import is necessary for other imports:
from pydtnn.lr_schedulers.early_stopping import EarlyStopping as _EarlyStopping
from pydtnn.lr_schedulers.model_checkpoint import ModelCheckpoint as _ModelCheckpoint
from pydtnn.lr_schedulers.reduce_lr_every_n_epochs import ReduceLREveryNEpochs as _ReduceLREveryNEpochs
from pydtnn.lr_schedulers.reduce_lr_on_plateau import ReduceLROnPlateau as _ReduceLROnPlateau
from pydtnn.lr_schedulers.stop_at_loss import StopAtLoss as _StopAtLoss
from pydtnn.lr_schedulers.warm_up_lr_scheduler import WarmUpLRScheduler as _WarmUpLRScheduler


def get_lr_schedulers(model) -> list[_LRScheduler]:
    """Get LR Scheduler objects from model attributes"""
    lr_schedulers = []
    # NOTE: All this parameters came from Parser
    for lr_sched in filter(None, model.lr_schedulers_names.split(",")):
        match lr_sched:
            case "warm_up":
                lrs = _WarmUpLRScheduler(model.warm_up_epochs,
                                         model.learning_rate / model.nprocs,
                                         model.learning_rate)
            case "early_stopping":
                lrs = _EarlyStopping(model.early_stopping_metric,
                                     model.early_stopping_patience,
                                     model.early_stopping_minimize)
            case "reduce_lr_on_plateau":
                lrs = _ReduceLROnPlateau(model.reduce_lr_on_plateau_metric,
                                         model.reduce_lr_on_plateau_factor,
                                         model.reduce_lr_on_plateau_patience,
                                         model.reduce_lr_on_plateau_min_lr)
            case "reduce_lr_every_nepochs":
                lrs = _ReduceLREveryNEpochs(model.reduce_lr_every_nepochs_factor,
                                            model.reduce_lr_every_nepochs_nepochs,
                                            model.reduce_lr_every_nepochs_min_lr)
            case "stop_at_loss":
                lrs = _StopAtLoss(model.stop_at_loss_metric,
                                  model.stop_at_loss_threshold)
            case "model_checkpoint":
                lrs = _ModelCheckpoint(model.model_checkpoint_metric,
                                       model.model_checkpoint_save_freq)
            case _:
                raise SystemExit(f"LRScheduler '{model.optimizer}' not supported yet!")
        lr_schedulers.append(lrs)
    return lr_schedulers
