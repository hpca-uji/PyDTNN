"""Get a Resnet50 from PyTorch converted to PyDTNN."""

from pydtnn.models.resnet50_converter_pytorch import resnet50_converter_pytorch as resnet50_from_pytorch

__all__ = ("resnet50_from_pytorch",)
