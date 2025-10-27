# _______________________________________________________________________________________________________________
# In this file must be implemented only the translation of PyTorch Dropout layers to its PyDTNN equivalent.
# _______________________________________________________________________________________________________________

# Typing related (or non important) imports
from typing import Dict, Any

# Functionality imports
from pydtnn.layers.dropout import Dropout as _Dropout
import pydtnn.converters.pytorch2pydtnn.common as cm

# ------------------- #


def Dropout(args: Dict[str, Any]) -> _Dropout:
    # https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html#torch.nn.Dropout

    # PyTorch attributes:
    # Not used: inplace: Bool
    PYTORCH_P = "p"
    torch_dict_keys = [PYTORCH_P]
    # ---- #

    # PyDTNN attributes:
    PYDTNN_RATE = "rate"
    pydtnn_dict_keys = [PYDTNN_RATE]
    # ---- #

    layer_args = cm.prepare_pydtnn_arguments(arguments=args[cm.ARGUMENTS], torch_dict_keys=torch_dict_keys, pydtnn_dict_keys=pydtnn_dict_keys)

    return _Dropout(**layer_args)
