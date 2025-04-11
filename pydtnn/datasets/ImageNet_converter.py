#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021-22 Universitat Jaume I
#
#  PyDTNN is free software: you can redistribute it and/or modify it under the
#  terms of the GNU General Public License as published by the Free Software
#  Foundation, either version 3 of the License, or (at your option) any later
#  version.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT
#  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
#  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
#  License for more details.
#
#  You should have received a copy of the GNU General Public License along
#  with this program. If not, see <https://www.gnu.org/licenses/>.
#

import argparse

import numpy as np
import tensorflow as tf


def distort_color(image, scope=None):
    """Distort the color of the image.
    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather than adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.
    Args:
      image: Tensor containing single image.
      scope: Optional scope for name_scope.
    Returns:
      color-distorted image
    """
    with tf.name_scope(scope or "distort_color"):
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image


def distort_image(image, height, width, bbox, scope=None):
    """Distort one image for training a network.
    Distorting images provides a useful technique for augmenting the data
    set during training in order to make the network invariant to aspects
    of the image that do not effect the label.
    Args:
      image: 3-D float Tensor of image
      height: integer
      width: integer
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged
        as [ymin, xmin, ymax, xmax].
      scope: Optional scope for name_scope.
    Returns:
      3-D float Tensor of distorted image used for training.
    """
    with tf.name_scope(scope or "distort_image"):
        # Show original bounding box
        # image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0), bbox)
        # tf.summary.image("image_with_bounding_boxes", image_with_box)

        bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=bbox,
            min_object_covered=0.1,
            aspect_ratio_range=(0.75, 1.33),
            area_range=(0.05, 1.0),
            max_attempts=100,
            use_image_if_no_bounding_boxes=True
        )

        # Show distorted bounding box
        # image_with_distorted_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0), distort_bbox)
        # tf.summary.image("images_with_distorted_bounding_box", image_with_distorted_box)

        # Crop image using distorted bounding box
        distorted_image = tf.slice(image, bbox_begin, bbox_size)

        # Resize image
        distorted_image = tf.image.resize(distorted_image, [height, width], method=tf.image.ResizeMethod.BILINEAR)
        distorted_image.set_shape([height, width, 3])

        # Show cropped image
        # tf.summary.image("cropped_resized_image", tf.expand_dims(distorted_image, 0))

        # Randomly flip image
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        # Distort color
        distorted_image = distort_color(distorted_image)

        # Show distorted image
        # tf.summary.image("final_distorted_image", tf.expand_dims(distorted_image, 0))
        return distorted_image


def decode_jpeg(image_buffer, scope=None):
    """Decode a JPEG string into one 3-D float image Tensor.
    Args:
      image_buffer: scalar string Tensor.
      scope: Optional scope for name_scope.
    Returns:
      3-D float Tensor with values ranging from [0, 1).
    """
    with tf.name_scope(scope or "decode_jpeg"):
        image = tf.image.decode_jpeg(image_buffer, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image


def eval_image(image, height, width, scope=None):
    """Proprocessing for evaluation (crop and scale)"""
    with tf.name_scope(scope or "eval_image"):
        image = tf.image.central_crop(image, central_fraction=0.875)
        image = tf.expand_dims(image, 0)
        image = tf.image.resize(image, [height, width], method=tf.image.ResizeMethod.BILINEAR)
        image = tf.squeeze(image, axis=0)
        return image


def image_processing(image_buffer, size=300):
    """Process image buffer"""
    image = decode_jpeg(image_buffer)
    width = height = size

    # Distort image
    # image = distort_image(image, height, width, bbox)

    image = eval_image(image, height, width)

    # Remap to [-1, 1]
    # image = tf.subtract(image, 0.5)
    # image = tf.multiply(image, 2.0)
    return image


def _parse_function(example_serialized):
    """Parse example from serialized TFRecord"""
    sparse_float32 = tf.io.VarLenFeature(dtype=tf.float32)

    # Dense features in Example proto
    feature_map = {
        "image/filename": tf.io.FixedLenFeature([], dtype=tf.string, default_value=""),
        "image/encoded": tf.io.FixedLenFeature([], dtype=tf.string, default_value=""),
        "image/class/label": tf.io.FixedLenFeature([1], dtype=tf.int64, default_value=[-1]),
        "image/class/text": tf.io.FixedLenFeature([], dtype=tf.string, default_value=""),
    }

    # Sparse features in Example proto
    feature_map.update({
        k: sparse_float32 for k in [
            "image/object/bbox/xmin",
            "image/object/bbox/ymin",
            "image/object/bbox/xmax",
            "image/object/bbox/ymax"
        ]
    })

    features = tf.io.parse_single_example(example_serialized, feature_map)
    label = tf.cast(features["image/class/label"], dtype=tf.int32)

    xmin = tf.expand_dims(features["image/object/bbox/xmin"].values, 0)
    ymin = tf.expand_dims(features["image/object/bbox/ymin"].values, 0)
    xmax = tf.expand_dims(features["image/object/bbox/xmax"].values, 0)
    ymax = tf.expand_dims(features["image/object/bbox/ymax"].values, 0)

    # Note that we impose an ordering of (y, x) just to make life difficult
    bbox = tf.concat([ymin, xmin, ymax, xmax], axis=0)

    # Force the variable number of bounding boxes into the shape
    # [1, num_boxes, coords]
    bbox = tf.expand_dims(bbox, 0)
    bbox = tf.transpose(bbox, perm=[0, 2, 1])

    return features["image/encoded"], label, bbox, features["image/class/text"]


def load_tfrecords(src, size):
    """Load a TFRecord and return images and labels as NumPy arrays"""
    dataset = tf.data.TFRecordDataset(src)
    dataset = dataset.map(_parse_function)
    # dataset = dataset.repeat(2) # repeat for 2 epochs
    # dataset = dataset.batch(5) # set batch_size = 5

    images_list = []
    labels_list = []

    for image_buffer, label_index, bbox, _ in dataset:
        image = image_processing(image_buffer, size=size)

        # Convert image (float [0,1) to uint8)
        image_np = (image.numpy() * 255).astype(np.uint8)
        images_list.append(image_np)
        labels_list.append(label_index.numpy())

    # Stack images in HWCN format
    images_np = np.stack(images_list, axis=-1)
    labels_np = np.array(labels_list)
    return images_np, labels_np


def main():
    """Application entrypoint"""
    parser = argparse.ArgumentParser(description="Convert TFRecords to NPZs")
    parser.add_argument("start", type=int, default=0, help="Start file index")
    parser.add_argument("end", type=int, default=1024, help="End file index")
    parser.add_argument("--size", type=int, default=300, help="Desired image size")
    parser.add_argument("--src", type=str, default="datasets/imagenet/tf/data/train/%05d.tfrecord", help="Source file path")
    parser.add_argument("--dst", type=str, default="datasets/imagenet/train/%05d.npz", help="Destination file path")
    args = parser.parse_args()

    for file in range(args.start, args.end):
        tfrecord_file = args.src % file
        npz_file = args.dst % file

        print(f"Processing: {tfrecord_file}")
        x, y = load_tfrecords(tfrecord_file, args.size)

        # Transpose to NCHW format
        x = np.transpose(x, (3, 2, 0, 1))

        print(f"Saving: {npz_file}")
        np.savez_compressed(npz_file, x=x, y=y)


if __name__ == "__main__":
    main()