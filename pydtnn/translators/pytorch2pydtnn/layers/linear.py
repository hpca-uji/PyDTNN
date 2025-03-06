# _______________________________________________________________________________________________________________
# In this file must be implemented only the translation of PyTorch Convolutional layers to its PyDTNN equivalent.
# _______________________________________________________________________________________________________________

# Typing related (or non important) imports
from typing import *
from inspect import stack # This is only in order to get the function's name

# Functionality imports
from pydtnn import layers
import pydtnn.translators.pytorch2pydtnn.constats as cons


# ------------------ #
def Linear(args: Dict[str, Any]) -> layers.FC:
    # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear
        
    print(f"Layer: {stack()[0].function}")
    # PyTorch attributes:
    # Not used: in_features, out_features
    PYTORCH_BIAS = "bias"
    torch_dict_keys = [PYTORCH_BIAS]
    # ---- #

    # PyDTNN attributes:
    PYDTNN_BIAS = "bias"
    pydtnn_dict_keys = [PYDTNN_BIAS]
    # ---- #   

    layer_args = cons.prepare_pydtnn_arguments(arguments = args[cons.ARGUMENTS], torch_dict_keys = torch_dict_keys, pydtnn_dict_keys = pydtnn_dict_keys)

    return layers.FC(**layer_args)
# --- END Linear --- #