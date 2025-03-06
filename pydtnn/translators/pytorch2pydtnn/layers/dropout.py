# _______________________________________________________________________________________________________________
# In this file must be implemented only the translation of PyTorch Convolutional layers to its PyDTNN equivalent.
# _______________________________________________________________________________________________________________

# Typing related (or non important) imports
from typing import *
from inspect import stack # This is only in order to get the function's name

# Functionality imports
from pydtnn import layers
import pydtnn.translators.pytorch2pydtnn.constats as cons

# ------------------- #
def Dropout(args: Dict[str, Any]) -> layers.Dropout:
    # https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html#torch.nn.Dropout
    
    print(f"Layer: {stack()[0].function}")
    # PyTorch attributes:
    # Not used: inplace: Bool
    PYTORCH_P = "p"
    torch_dict_keys = [PYTORCH_P]
    # ---- #

    # PyDTNN attributes:
    PYDTNN_RATE = "rate"
    pydtnn_dict_keys = [PYDTNN_RATE]
    # ---- #   

    layer_args = cons.prepare_pydtnn_arguments(arguments = args[cons.ARGUMENTS], torch_dict_keys = torch_dict_keys, pydtnn_dict_keys = pydtnn_dict_keys)

    return layers.Dropout(**layer_args)
# --- END Dropout --- #