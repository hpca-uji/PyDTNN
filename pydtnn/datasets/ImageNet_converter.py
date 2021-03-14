#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021 Universitat Jaume I
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

import sys

import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('image_size', 300,
                            """Provide square images of this size.""")
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            """Number of preprocessing threads per tower. """
                            """Please make this a multiple of 4.""")
tf.app.flags.DEFINE_integer('num_readers', 4,
                            """Number of parallel readers during train.""")


def distort_color(image, thread_id=0, scope=None):
    """Distort the color of the image.
    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather than adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.
    Args:
      image: Tensor containing single image.
      thread_id: preprocessing thread ID.
      scope: Optional scope for name_scope.
    Returns:
      color-distorted image
    """
    with tf.name_scope(values=[image], name=scope, default_name='distort_color'):
        color_ordering = thread_id % 2

        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)

        # The random_* ops do not necessarily clamp.
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image


def distort_image(image, height, width, bbox, thread_id=0, scope=None):
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
      thread_id: integer indicating the preprocessing thread.
      scope: Optional scope for name_scope.
    Returns:
      3-D float Tensor of distorted image used for training.
    """
    with tf.name_scope(values=[image, height, width, bbox], name=scope,
                       default_name='distort_image'):
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].

        # Display the bounding box in the first thread only.
        if not thread_id:
            image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
                                                          bbox)
            tf.summary.image('image_with_bounding_boxes', image_with_box)

        # A large fraction of image datasets contain a human-annotated bounding
        # box delineating the region of the image containing the object of interest.
        # We choose to create a new bounding box for the object which is a randomly
        # distorted version of the human-annotated bounding box that obeys an allowed
        # range of aspect ratios, sizes and overlap with the human-annotated
        # bounding box. If no box is supplied, then we assume the bounding box is
        # the entire image.
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=bbox,
            min_object_covered=0.1,
            aspect_ratio_range=[0.75, 1.33],
            area_range=[0.05, 1.0],
            max_attempts=100,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box
        if not thread_id:
            image_with_distorted_box = tf.image.draw_bounding_boxes(
                tf.expand_dims(image, 0), distort_bbox)
            tf.summary.image('images_with_distorted_bounding_box',
                             image_with_distorted_box)

        # Crop the image to the specified bounding box.
        distorted_image = tf.slice(image, bbox_begin, bbox_size)

        # This resizing operation may distort the images because the aspect
        # ratio is not respected. We select a resize method in a round robin
        # fashion based on the thread number.
        # Note that ResizeMethod contains 4 enumerated resizing methods.
        resize_method = thread_id % 4
        distorted_image = tf.image.resize_images(distorted_image, [height, width],
                                                 method=resize_method)
        # Restore the shape since the dynamic slice based upon the bbox_size loses
        # the third dimension.
        distorted_image.set_shape([height, width, 3])
        if not thread_id:
            tf.summary.image('cropped_resized_image',
                             tf.expand_dims(distorted_image, 0))

        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        # Randomly distort the colors.
        distorted_image = distort_color(distorted_image, thread_id)

        if not thread_id:
            tf.summary.image('final_distorted_image',
                             tf.expand_dims(distorted_image, 0))
        return distorted_image


def decode_jpeg(image_buffer, scope=None):
    """Decode a JPEG string into one 3-D float image Tensor.
    Args:
      image_buffer: scalar string Tensor.
      scope: Optional scope for name_scope.
    Returns:
      3-D float Tensor with values ranging from [0, 1).
    """
    with tf.name_scope(values=[image_buffer], name=scope,
                       default_name='decode_jpeg'):
        # Decode the string as an RGB JPEG.
        # Note that the resulting image contains an unknown height and width
        # that is set dynamically by decode_jpeg. In other words, the height
        # and width of image is unknown at compile-time.
        image = tf.image.decode_jpeg(image_buffer, channels=3)

        # After this point, all image pixels reside in [0,1)
        # until the very end, when they're rescaled to (-1, 1).  The various
        # adjust_* ops all require this range for dtype float.
        image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
        return image


def eval_image(image, height, width, scope=None):
    with tf.name_scope(values=[image, height, width], name=scope,
                       default_name='eval_image'):
        # Crop the central region of the image with an area containing 87.5% of
        # the original image.
        image = tf.image.central_crop(image, central_fraction=0.875)

        # Resize the image to the original height and width.
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [height, width],
                                         align_corners=False)
        image = tf.squeeze(image, [0])
        return image


def image_preprocessing(image_buffer, bbox, thread_id=0):
    if bbox is None:
        raise ValueError('Please supply a bounding box.')

    image = decode_jpeg(image_buffer)
    height = FLAGS.image_size
    width = FLAGS.image_size

    # if train:
    # image = distort_image(image, height, width, bbox, thread_id)
    # else:
    image = eval_image(image, height, width)

    # Finally, rescale to [-1,1] instead of [0, 1)
    # image = tf.subtract(image, 0.5)
    # image = tf.multiply(image, 2.0)
    return image


def _parse_function(example_serialized):
    # Dense features in Example proto.
    feature_map = {
        'image/filename': tf.io.FixedLenFeature([], dtype=tf.string,
                                                default_value=''),
        'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string,
                                               default_value=''),
        'image/class/label': tf.io.FixedLenFeature([1], dtype=tf.int64,
                                                   default_value=-1),
        'image/class/text': tf.io.FixedLenFeature([], dtype=tf.string,
                                                  default_value=''),
    }
    sparse_float32 = tf.io.VarLenFeature(dtype=tf.float32)
    # Sparse features in Example proto.
    feature_map.update(
        {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                     'image/object/bbox/ymin',
                                     'image/object/bbox/xmax',
                                     'image/object/bbox/ymax']})

    features = tf.io.parse_single_example(example_serialized, feature_map)
    label = tf.cast(features['image/class/label'], dtype=tf.int32)

    xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
    xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
    ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

    # Note that we impose an ordering of (y, x) just to make life difficult.
    bbox = tf.concat(axis=0, values=[ymin, xmin, ymax, xmax])

    # Force the variable number of bounding boxes into the shape
    # [1, num_boxes, coords].
    bbox = tf.expand_dims(bbox, 0)
    bbox = tf.transpose(bbox, [0, 2, 1])

    return features['image/encoded'], label, bbox, features['image/class/text']


def load_tfrecords(srcfile, image_size, elems):
    sess = tf.compat.v1.Session()

    FLAGS.image_size = image_size
    dataset = tf.data.TFRecordDataset(srcfile)  # load tfrecord file
    dataset = dataset.map(_parse_function)  # parse data into tensor
    # dataset = dataset.repeat(2) # repeat for 2 epochs
    # dataset = dataset.batch(1) # set batch_size = 5

    iterator = dataset.make_one_shot_iterator()
    next_data = iterator.get_next()

    images = []
    labels = []

    count = 0
    while count < elems:
        try:
            image_buffer, label_index, bbox, _ = sess.run(next_data)
            image = image_preprocessing(image_buffer, bbox)
            if count == 10:
                break
            # print(count)
            if not images:
                images = np.expand_dims(np.array(image.eval(session=sess)).astype(np.uint8), axis=-1)
            else:
                images = np.concatenate(
                    (images, np.expand_dims(np.array(image.eval(session=sess)).astype(np.uint8), axis=-1)), axis=-1)
            labels.extend([label_index])

            # plt.imshow(data, interpolation='nearest')
            # plt.show()
        except tf.errors.OutOfRangeError:
            break
        count += 1

    labels = np.array(labels)
    return images, labels


if __name__ == "__main__":

    start, end = int(sys.argv[1]), int(sys.argv[2])
    for file in range(start, end):
        x, y = load_tfrecords("/scratch/imagenet/train/train-%05d-01024" % file, 227, 10000)
        np.savez_compressed("/scratch/imagenet/np/train/train-%05d-01024" % file, x=x.transpose((3, 2, 0, 1)), y=y)
