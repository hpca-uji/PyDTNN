"""
ResNet34 model architecture adapted for CIFAR-10 dataset.
"""
from pydtnn.models.resnet34 import resnet34 as resnet34_cifar10

__all__ = ("resnet34_cifar10",)

def resnet34_cifar10():
    """
    Constructs a ResNet34 model configured for CIFAR-10 input dimensions.

    Returns:
        nn.Module: The ResNet34 model instance.
    """
    return resnet34_cifar10()