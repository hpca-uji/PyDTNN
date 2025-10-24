# _______________________________________________________________________________________________________________
# In this file must be implemented only the translation of PyTorch Convolutional layers to its PyDTNN equivalent.
# _______________________________________________________________________________________________________________

# Typing related (or non important) imports
from typing import Dict

# Functionality imports
from pydtnn.layers.flatten import Flatten as _Flatten
# import pydtnn.converters.pytorch2pydtnn.common as cm

# ------------------ #


def Flatten(args: Dict[str, str]) -> _Flatten:
    # https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html#torch.nn.Flatten
    # torch.nn.Flatten(start_dim=1, end_dim=-1)

    # PyTorch attributes:
    # Not used: start_dim, end_dim (It's not used due the way the layer's initialization works in PyDTNN)
    # ---- #
    # PyDTNN attributes: None
    # ---- #
    not_used = args

    return _Flatten()
# ------------------ #
