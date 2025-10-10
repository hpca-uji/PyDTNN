import io
import sys
import enum
import typing
import tarfile
import argparse
import itertools
from pathlib import Path
from collections import abc

import numpy as np
from PIL import Image
from scipy.io import loadmat


class Mode(enum.StrEnum):
    """Dataset type"""
    TRAIN = enum.auto()
    VAL = enum.auto()


parser = argparse.ArgumentParser(description="Convert dataset to NPZs")
parser.add_argument("--mode", type=Mode, choices=list(Mode), default=Mode.TRAIN, help="dateset type")
parser.add_argument("--meta", type=Path, default=Path("datasets/imagenet/ILSVRC2012_devkit_t12.tar.gz"), help="metadata path")
parser.add_argument("--src", type=Path, default=Path("datasets/imagenet/ILSVRC2012_img_train.tar"), help="source path")
parser.add_argument("--dst", type=Path, default=Path("datasets/imagenet/train/%05d.npz"), help="destination path")
parser.add_argument("--crop", type=float, default=0.875, help="image crop")
parser.add_argument("--size", type=int, default=227, help="image size")
parser.add_argument("--batch", type=int, default=1251, help="batch size")


def central_crop(image: Image.Image, central_fraction=0.875) -> Image.Image:
    """Crop central fraction of an image"""
    width, height = image.size
    frame_fraction = (1 - central_fraction) / 2
    x_offset, y_offset = width * frame_fraction, height * frame_fraction
    return image.crop((x_offset, y_offset, width - x_offset, height - y_offset))


def load_tar(fp: typing.IO[bytes]) -> abc.Generator[typing.IO[bytes]]:
    """Generate tar-file members"""
    with tarfile.TarFile(fileobj=fp) as group:
        for member in group.getmembers():
            member_fp = group.extractfile(member)
            if member_fp is None:
                continue
            with member_fp:
                yield member_fp


def load_train_label(fp: typing.IO[bytes], labels: dict[int, int]) -> int:
    """Transform a file-like (image) to int (label)"""
    label = fp.name
    label = label.replace("_", ".")
    label = label.split(".")[0]
    label = label.lstrip("n")
    label = int(label)
    return labels[label]


def load_val_label(fp: typing.IO[bytes], labels: dict[int, int]) -> int:
    """Transform a file-like (image) to int (label)"""
    label = fp.name
    label = label.replace("_", ".")
    label = label.split(".")[2]
    label = int(label)
    return labels[label]


def load_image(fp: typing.IO[bytes], crop: float = 0.875, size: int = 64) -> np.ndarray:
    """Transform a file-like (image) to ndarray (data)"""
    with Image.open(fp=fp) as image:
        image = image.convert("RGB")
        image = central_crop(image, crop)
        image = image.resize((size, size))
        array = np.asarray(image)
        array = array.transpose((2, 0, 1))
        array = array.copy()
    return array


def get_train_labels(config: argparse.Namespace) -> dict[int, int]:
    """Get label mappings from archive"""
    with tarfile.open(config.meta) as fp:
        with fp.extractfile("ILSVRC2012_devkit_t12/data/meta.mat") as fp:
            meta = loadmat(file_name=fp, squeeze_me=True)["synsets"]
    nums_children = list(zip(*meta))[4]
    meta = [meta[idx] for idx, num_children in enumerate(nums_children) if num_children == 0]
    labels, codes, class_name = list(zip(*meta))[:3]
    return {
        int(code.lstrip("n")): int(label)
        for code, label in zip(codes, labels)
    }


def get_val_labels(config: argparse.Namespace) -> dict[int, int]:
    """Get label mappings from archive"""
    with tarfile.open(config.meta) as fp:
        with fp.extractfile("ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt") as fp:
            with io.TextIOWrapper(buffer=fp) as fp:
                return {
                    i: int(line)
                    for i, line in enumerate(fp, 1)
                }


def load_train(path: Path, config: argparse.Namespace) -> abc.Generator[tuple[int, np.ndarray]]:
    """Load labeled images from training archive"""
    labels = get_train_labels(config)
    with open(path, mode="rb") as fp:
        for fp in load_tar(fp=fp):
            for fp in load_tar(fp=fp):
                label = load_train_label(fp=fp, labels=labels)
                image = load_image(fp=fp, crop=config.crop, size=config.size)
                yield label, image


def load_val(path: Path, config: argparse.Namespace) -> abc.Generator[tuple[int, np.ndarray]]:
    """Load labeled images from valing archive"""
    labels = get_val_labels(config)
    with open(path, mode="rb") as fp:
        for fp in load_tar(fp=fp):
            label = load_val_label(fp=fp, labels=labels)
            image = load_image(fp=fp, crop=config.crop, size=config.size)
            yield label, image


def main(config: argparse.Namespace) -> None:
    """Application entrypoint"""
    module = sys.modules[__name__]
    loader = getattr(module, f"load_{config.mode}")
    labeled_images = loader(path=config.src, config=config)

    for i, group in enumerate(itertools.batched(labeled_images, n=config.batch)):
        path = Path(str(config.dst) % i)
        labels, images = zip(*group)
        labels = np.array(labels, dtype=np.int32)
        images = np.stack(images, dtype=np.uint8)
        np.savez_compressed(path, x=images, y=labels)
        print(f"Saved: {path}, x={images.dtype}{images.shape}, y={labels.dtype}{labels.shape}")


if __name__ == "__main__":
    main(parser.parse_args())
