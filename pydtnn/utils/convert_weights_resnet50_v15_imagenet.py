"""
PyDTNN: TF to PyDTNN ResNet-50-v1.5 weights converter
"""

import os
import sys
import h5py
import urllib.request
import numpy as np
from pydtnn.utils.constants import Parameters

# PyDTNN <-> Keras layer conversion
layers = {"1_Conv2D": "conv1_conv",
          "2_BatchNormalization": "conv1_bn",
          "6_Conv2D": "conv2_block1_1_conv",
          "7_BatchNormalization": "conv2_block1_1_bn",
          "9_Conv2D": "conv2_block1_2_conv",
          "10_BatchNormalization": "conv2_block1_2_bn",
          "12_Conv2D": "conv2_block1_3_conv",
          "13_BatchNormalization": "conv2_block1_3_bn",
          "14_Conv2D": "conv2_block1_0_conv",
          "15_BatchNormalization": "conv2_block1_0_bn",
          "18_Conv2D": "conv2_block2_1_conv",
          "19_BatchNormalization": "conv2_block2_1_bn",
          "21_Conv2D": "conv2_block2_2_conv",
          "22_BatchNormalization": "conv2_block2_2_bn",
          "24_Conv2D": "conv2_block2_3_conv",
          "25_BatchNormalization": "conv2_block2_3_bn",
          "28_Conv2D": "conv2_block3_1_conv",
          "29_BatchNormalization": "conv2_block3_1_bn",
          "31_Conv2D": "conv2_block3_2_conv",
          "32_BatchNormalization": "conv2_block3_2_bn",
          "34_Conv2D": "conv2_block3_3_conv",
          "35_BatchNormalization": "conv2_block3_3_bn",
          "38_Conv2D": "conv3_block1_1_conv",
          "39_BatchNormalization": "conv3_block1_1_bn",
          "41_Conv2D": "conv3_block1_2_conv",
          "42_BatchNormalization": "conv3_block1_2_bn",
          "44_Conv2D": "conv3_block1_3_conv",
          "45_BatchNormalization": "conv3_block1_3_bn",
          "46_Conv2D": "conv3_block1_0_conv",
          "47_BatchNormalization": "conv3_block1_0_bn",
          "50_Conv2D": "conv3_block2_1_conv",
          "51_BatchNormalization": "conv3_block2_1_bn",
          "53_Conv2D": "conv3_block2_2_conv",
          "54_BatchNormalization": "conv3_block2_2_bn",
          "56_Conv2D": "conv3_block2_3_conv",
          "57_BatchNormalization": "conv3_block2_3_bn",
          "60_Conv2D": "conv3_block3_1_conv",
          "61_BatchNormalization": "conv3_block3_1_bn",
          "63_Conv2D": "conv3_block3_2_conv",
          "64_BatchNormalization": "conv3_block3_2_bn",
          "66_Conv2D": "conv3_block3_3_conv",
          "67_BatchNormalization": "conv3_block3_3_bn",
          "70_Conv2D": "conv3_block4_1_conv",
          "71_BatchNormalization": "conv3_block4_1_bn",
          "73_Conv2D": "conv3_block4_2_conv",
          "74_BatchNormalization": "conv3_block4_2_bn",
          "76_Conv2D": "conv3_block4_3_conv",
          "77_BatchNormalization": "conv3_block4_3_bn",
          "80_Conv2D": "conv4_block1_1_conv",
          "81_BatchNormalization": "conv4_block1_1_bn",
          "83_Conv2D": "conv4_block1_2_conv",
          "84_BatchNormalization": "conv4_block1_2_bn",
          "86_Conv2D": "conv4_block1_3_conv",
          "87_BatchNormalization": "conv4_block1_3_bn",
          "88_Conv2D": "conv4_block1_0_conv",
          "89_BatchNormalization": "conv4_block1_0_bn",
          "92_Conv2D": "conv4_block2_1_conv",
          "93_BatchNormalization": "conv4_block2_1_bn",
          "95_Conv2D": "conv4_block2_2_conv",
          "96_BatchNormalization": "conv4_block2_2_bn",
          "98_Conv2D": "conv4_block2_3_conv",
          "99_BatchNormalization": "conv4_block2_3_bn",
          "102_Conv2D": "conv4_block3_1_conv",
          "103_BatchNormalization": "conv4_block3_1_bn",
          "105_Conv2D": "conv4_block3_2_conv",
          "106_BatchNormalization": "conv4_block3_2_bn",
          "108_Conv2D": "conv4_block3_3_conv",
          "109_BatchNormalization": "conv4_block3_3_bn",
          "112_Conv2D": "conv4_block4_1_conv",
          "113_BatchNormalization": "conv4_block4_1_bn",
          "115_Conv2D": "conv4_block4_2_conv",
          "116_BatchNormalization": "conv4_block4_2_bn",
          "118_Conv2D": "conv4_block4_3_conv",
          "119_BatchNormalization": "conv4_block4_3_bn",
          "122_Conv2D": "conv4_block5_1_conv",
          "123_BatchNormalization": "conv4_block5_1_bn",
          "125_Conv2D": "conv4_block5_2_conv",
          "126_BatchNormalization": "conv4_block5_2_bn",
          "128_Conv2D": "conv4_block5_3_conv",
          "129_BatchNormalization": "conv4_block5_3_bn",
          "132_Conv2D": "conv4_block6_1_conv",
          "133_BatchNormalization": "conv4_block6_1_bn",
          "135_Conv2D": "conv4_block6_2_conv",
          "136_BatchNormalization": "conv4_block6_2_bn",
          "138_Conv2D": "conv4_block6_3_conv",
          "139_BatchNormalization": "conv4_block6_3_bn",
          "142_Conv2D": "conv5_block1_1_conv",
          "143_BatchNormalization": "conv5_block1_1_bn",
          "145_Conv2D": "conv5_block1_2_conv",
          "146_BatchNormalization": "conv5_block1_2_bn",
          "148_Conv2D": "conv5_block1_3_conv",
          "149_BatchNormalization": "conv5_block1_3_bn",
          "150_Conv2D": "conv5_block1_0_conv",
          "151_BatchNormalization": "conv5_block1_0_bn",
          "154_Conv2D": "conv5_block2_1_conv",
          "155_BatchNormalization": "conv5_block2_1_bn",
          "157_Conv2D": "conv5_block2_2_conv",
          "158_BatchNormalization": "conv5_block2_2_bn",
          "160_Conv2D": "conv5_block2_3_conv",
          "161_BatchNormalization": "conv5_block2_3_bn",
          "164_Conv2D": "conv5_block3_1_conv",
          "165_BatchNormalization": "conv5_block3_1_bn",
          "167_Conv2D": "conv5_block3_2_conv",
          "168_BatchNormalization": "conv5_block3_2_bn",
          "170_Conv2D": "conv5_block3_3_conv",
          "171_BatchNormalization": "conv5_block3_3_bn",
          "175_FC": "probs"}

# PyDTNN <-> Keras weights name conversion
weights = {"kernel": Parameters.WEIGHTS,
           "bias": Parameters.BIASES,
           Parameters.GAMMA: Parameters.GAMMA,
           Parameters.BETA: Parameters.BETA,
           "moving_mean": "running_mean",
           "moving_variance": "running_var"}


# ResNet-50-v1.5 HDF5 file
IN_FILE = 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'
OUT_FILE = 'resnet50_weights_pydtnn_kernels.npz'
PATH = 'https://storage.googleapis.com/tensorflow/keras-applications/resnet/'

if __name__ == "__main__":
    # Download weights file
    if not os.path.exists(IN_FILE):
        try:
            urllib.request.urlretrieve(PATH + IN_FILE, IN_FILE)
        except BaseException as exc:
            raise SystemExit(f"Error while downloading {PATH + IN_FILE}") from exc

    f = h5py.File("resnet50_weights_tf_dim_ordering_tf_kernels.h5", "r")
    out = {}
    for pydtnn_label, keras_label in layers.items():
        for weight_label in f[keras_label].attrs["weight_names"]:
            w = weight_label.decode("utf-8").split("/")[1].split(":")[0]
            w = weights[w]
            value = f[keras_label][weight_label][:]
            if value.ndim == 4:
                value = value.transpose(3, 2, 0, 1)
            out[f"{pydtnn_label}_{w}"] = value
    f.close()
    np.savez_compressed(OUT_FILE.split(".")[0], **out)

    print(f"Successfully written PyDTNN output file {OUT_FILE}!")
