from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydtnn.model import Model as _Model
    from pydtnn.schedulers.scheduler import Scheduler as _LRScheduler


def select(model: "_Model") -> "list[_LRScheduler]":
    """Get Scheduler objects from model attributes"""
    from pydtnn.schedulers.early_stopping import EarlyStopping
    from pydtnn.schedulers.model_checkpoint import ModelCheckpoint
    from pydtnn.schedulers.reduce_lr_every_n_epochs import ReduceLREveryNEpochs
    from pydtnn.schedulers.reduce_lr_on_plateau import ReduceLROnPlateau
    from pydtnn.schedulers.stop_at_loss import StopAtLoss
    from pydtnn.schedulers.warm_up_lr_scheduler import WarmUpLRScheduler

    schedulers = []
    for scheduler_name in filter(None, model.schedulers_names.split(",")):
        match scheduler_name:
            case "warm_up":
                scheduler = WarmUpLRScheduler(model.warm_up_epochs,
                                              model.learning_rate / model.nprocs,
                                              model.learning_rate)
            case "early_stopping":
                scheduler = EarlyStopping(model.early_stopping_metric,
                                          model.early_stopping_patience,
                                          model.early_stopping_minimize)
            case "reduce_lr_on_plateau":
                scheduler = ReduceLROnPlateau(model.reduce_lr_on_plateau_metric,
                                              model.reduce_lr_on_plateau_factor,
                                              model.reduce_lr_on_plateau_patience,
                                              model.reduce_lr_on_plateau_min_lr)
            case "reduce_lr_every_nepochs":
                scheduler = ReduceLREveryNEpochs(model.reduce_lr_every_nepochs_factor,
                                                 model.reduce_lr_every_nepochs_nepochs,
                                                 model.reduce_lr_every_nepochs_min_lr)
            case "stop_at_loss":
                scheduler = StopAtLoss(model.stop_at_loss_metric,
                                       model.stop_at_loss_threshold)
            case "model_checkpoint":
                scheduler = ModelCheckpoint(model.model_checkpoint_metric,
                                            model.model_checkpoint_save_freq)
            case _:
                raise SystemExit(f"LRScheduler '{model.optimizer}' not supported yet!")
        schedulers.append(scheduler)

    return schedulers
