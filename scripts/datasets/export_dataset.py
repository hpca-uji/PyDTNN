#!/usr/bin/env python3

"""
Exports a PyDTNN dataset.
"""

from pydtnn.model import Model
from pydtnn.utils.parser import PydtnnArgumentParser

parser = PydtnnArgumentParser()
parser.add_argument("--export-split-weights", type=str, default="")
model = Model(**parser.to_dict())
split_weights = list(map(float, filter(None, model.export_split_weights.split(","))))
model.dataset.export_archive(split_weights=split_weights)
