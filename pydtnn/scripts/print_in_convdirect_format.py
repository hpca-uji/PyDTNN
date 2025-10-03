#!/usr/bin/env python

"""
PyDTNN print in convdirect format script
"""

from pydtnn.model import Model
from pydtnn.parser import PydtnnArgumentParser

# Parse options
parser = PydtnnArgumentParser()
# Create model
model = Model(**parser.to_dict())
# Call print_in_convdirect_format
# print(f"#Model: {model.model_name}")
model.print_in_convdirect_format()

# Examples
# --------

# Print the layers of ResNet50v15 for ImageNet in convdirect input format:
# pydtnn/scripts/print_in_convdirect_format.py --model=resnet50v15_imagenet

# Print the memory required for the Im2Row transformation of each layer with the default batch size and float32
# pydtnn/scripts/print_in_convdirect_format.py --model=resnet50v15_imagenet \
#   | awk '!/#/ {print $1 " "  $5 * $6 * $7 * $8 * $9 * $10 * 4 / 1024 / 1024}' | sort -k 2  -g

# Print the memory required for the Im2Row transformation of each layer with batch size of 1 and float32
# pydtnn/scripts/print_in_convdirect_format.py --model=resnet50v15_imagenet \
#   | awk '!/#/ {print $1 " "  $6 * $7 * $8 * $9 * $10 * 4 / 1024 / 1024}' | sort -k 2  -g
