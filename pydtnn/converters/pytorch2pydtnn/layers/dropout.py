# In this file must be implemented only the translation of PyTorch Dropout layers to its PyDTNN equivalent.

import logging
from typing import Any

# Typing related (or non important) imports
import pydtnn.converters.pytorch2pydtnn.common as cm
from pydtnn.layers.dropout import Dropout as _Dropout

__all__ = ("Dropout",)

logger = logging.getLogger(__name__)


# Functionality imports


def Dropout(args: dict[str, Any]) -> _Dropout:
    # https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html#torch.nn.Dropout

    # PyTorch attributes:
    # Not used: inplace: Bool
    PYTORCH_P = "p"
    torch_dict_keys = [PYTORCH_P]

    # PyDTNN attributes:
    PYDTNN_RATE = "rate"
    pydtnn_dict_keys = [PYDTNN_RATE]

    layer_args = cm.prepare_pydtnn_arguments(arguments=args[cm.ARGUMENTS], torch_dict_keys=torch_dict_keys, pydtnn_dict_keys=pydtnn_dict_keys)

    return _Dropout(**layer_args)
