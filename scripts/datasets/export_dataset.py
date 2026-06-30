#!/usr/bin/env python3

"""Exports a PyDTNN dataset."""

from pydtnn.model import Model
from pydtnn.utils.parser import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--export-split-weights", type=str, default="")
model = Model(**vars(parser.parse_args()))
split_weights = list(map(float, filter(None, model.export_split_weights.split(","))))
model.dataset.export_archive(split_weights=split_weights)
