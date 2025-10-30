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
    from pydtnn.schedulers.warm_up_lr_scheduler import WarmUp

    schedulers = []
    for scheduler_name in filter(None, model.schedulers_names.split(",")):
        match scheduler_name:
            case "warm_up":
                scheduler = WarmUp.from_model(model)
            case "early_stopping":
                scheduler = EarlyStopping.from_model(model)
            case "reduce_lr_on_plateau":
                scheduler = ReduceLROnPlateau.from_model(model)
            case "reduce_lr_every_nepochs":
                scheduler = ReduceLREveryNEpochs.from_model(model)
            case "stop_at_loss":
                scheduler = StopAtLoss.from_model(model)
            case "model_checkpoint":
                scheduler = ModelCheckpoint.from_model(model)
            case _:
                raise SystemExit(f"LRScheduler '{model.optimizer}' not supported yet!")
        schedulers.append(scheduler)

    return schedulers
