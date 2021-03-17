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

import gc
import math
import os
import queue
import struct
import threading

import numpy as np


# @todo: split dataset.py into different files

class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, max_prefetch=1):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(max_prefetch)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


def do_flip_images(data, prob=0.5):
    n, c, h, w = data.shape
    limit = min(n, int(n * prob))
    s = np.arange(n)
    np.random.shuffle(s)
    s = s[:limit]
    data[s, ...] = np.flip(data[s, ...], axis=-1)
    return data


def do_crop_images(data, crop_size, prob=0.5):
    n, c, h, w = data.shape
    crop_size = min(crop_size, h, w)
    limit = min(n, int(n * prob))
    s = np.arange(n)
    np.random.shuffle(s)
    s = s[:limit]
    t = np.random.randint(0, h - crop_size, (limit,))
    ll = np.random.randint(0, w - crop_size, (limit,))
    for i, ri in enumerate(s):
        b, r = t[i] + crop_size, ll[i] + crop_size
        # batch[ri,...] = resize(batch[ri,:,t[i]:b,l[i]:r], (ri.size,c,h,w))
        data[ri, :, :t[i], :ll[i]] = 0.0
        data[ri, :, b:, r:] = 0.0
        data[ri, ...] = np.roll(data[ri, ...], np.random.randint(-t[i], (h - b)), axis=1)
        data[ri, ...] = np.roll(data[ri, ...], np.random.randint(-ll[i], (w - r)), axis=2)
    return data


class Dataset:

    def __init__(self, x_train=np.array([]), y_train=np.array([]),
                 x_val=np.array([]), y_val=np.array([]),
                 x_test=np.array([]), y_test=np.array([])):
        self.x_train, self.y_train = x_train, y_train
        self.x_val, self.y_val = x_val, y_val
        self.x_test, self.y_test = x_test, y_test
        self.train_val_nsamples = x_train.shape[0] + x_val.shape[0]
        self.train_nsamples = self.x_train.shape[0]
        self.test_as_validation = False
        # Attributes that will be properly defined elsewhere
        self.flip_images = None
        self.flip_images_prob = None
        self.crop_images = None
        self.crop_images_size = None
        self.crop_images_prob = None

    def make_train_val_partitions(self, val_split=0.2):
        pass

    def train_data_generator(self, batch_size):
        x_data, y_data = self.x_train, self.y_train
        # Use data augmentation, if used we preserve original data
        if self.flip_images or self.crop_images:
            x_data = x_data.copy()
        if self.flip_images:
            x_data = do_flip_images(x_data, self.flip_images_prob)
        if self.crop_images:
            x_data = do_crop_images(x_data, self.crop_images_size, self.crop_images_prob)
        yield x_data, y_data

    def val_data_generator(self, batch_size):
        yield self.x_val, self.y_val

    def test_data_generator(self, batch_size):
        yield self.x_test, self.y_test

    def get_train_val_generator(self, local_batch_size=64, rank=0, nprocs=1, val_split=0.2):
        batch_size = local_batch_size * nprocs
        return (self.batch_generator(self.train_data_generator(batch_size),
                                     local_batch_size, rank, nprocs, shuffle=True),
                self.batch_generator(self.val_data_generator(batch_size),
                                     local_batch_size, rank, nprocs, shuffle=False))

    def get_test_generator(self, local_batch_size=64, rank=0, nprocs=1):
        # Fixed batch size for testing:
        #   This is done to ensure that the returned x_data, y_data to the
        #   val_test_batch_generator will be larger enough to feed all processes.
        # local_batch_size = 64
        batch_size = local_batch_size * nprocs
        return self.batch_generator(self.test_data_generator(batch_size),
                                    local_batch_size, rank, nprocs, shuffle=False)

    @staticmethod
    def batch_generator(generator, local_batch_size=64, rank=0, nprocs=1, shuffle=True):
        batch_size = local_batch_size * nprocs

        for x_data, y_data in BackgroundGenerator(generator):
            nsamples = x_data.shape[0]
            s = memoryview(np.arange(nsamples))
            if shuffle:
                np.random.shuffle(s)

            last_batch_size = nsamples % batch_size
            if last_batch_size < nprocs:
                last_batch_size += batch_size
            end_for = nsamples - last_batch_size

            # Generate batches
            for batch_num in range(0, end_for, batch_size):
                start = batch_num + rank * local_batch_size
                end = batch_num + (rank + 1) * local_batch_size
                indices = s[start:end]
                x_local_batch = x_data[indices, ...]
                y_local_batch = y_data[indices, ...]
                yield x_local_batch, y_local_batch, batch_size

            # Generate last batch
            if last_batch_size > 0:
                last_local_batch_size = last_batch_size // nprocs
                remaining = last_batch_size % nprocs
                start = end = end_for
                if rank < remaining:
                    start += rank * (last_local_batch_size + 1)
                    end += (rank + 1) * (last_local_batch_size + 1)
                else:
                    start += remaining * (last_local_batch_size + 1) + (rank - remaining) * last_local_batch_size
                    end += remaining * (last_local_batch_size + 1) + (rank - remaining + 1) * last_local_batch_size
                indices = s[start:end]
                x_local_batch = x_data[indices, ...]
                y_local_batch = y_data[indices, ...]
                yield x_local_batch, y_local_batch, last_batch_size

    # def val_test_batch_generator(self, generator, rank=0, nprocs=1):
    #    for x_data, y_data in generator:
    #        batch_size = x_data.shape[0]
    #        local_batch_size = batch_size // nprocs
    #        remaining  = batch_size % nprocs
    #
    #        if rank < remaining:
    #           start =  rank    * (local_batch_size+1)
    #           end   = (rank+1) * (local_batch_size+1)
    #        else:
    #           start = remaining * (local_batch_size+1) + (rank-remaining) * local_batch_size
    #           end   = remaining * (local_batch_size+1) + (rank-remaining+1) * local_batch_size
    #
    #        x_local_batch = x_data[start:end,...]
    #        y_local_batch = y_data[start:end,...]
    #        yield (x_local_batch, y_local_batch, batch_size)


class MNIST(Dataset):

    def __init__(self, train_path, test_path, model="", test_as_validation=False,
                 flip_images=False, flip_images_prob=0.5,
                 crop_images=False, crop_images_size=14, crop_images_prob=0.5,
                 dtype=np.float32, use_synthetic_data=False):
        self.train_path = train_path
        self.test_path = test_path
        self.model = model
        self.test_as_validation = test_as_validation
        self.flip_images = flip_images
        self.flip_images_prob = flip_images_prob
        self.crop_images = crop_images
        self.crop_images_size = crop_images_size
        self.crop_images_prob = crop_images_prob
        self.dtype = dtype
        self.use_synthetic_data = use_synthetic_data
        self.nclasses = 10
        # self.val_start = 0

        self.train_val_nsamples = 60000
        self.test_nsamples = 10000
        self.shape = (1, 28, 28)

        self.val_start = np.random.randint(0, high=self.train_val_nsamples)

        if self.use_synthetic_data:
            self.x_train_val, self.y_train_val = \
                np.empty((self.train_val_nsamples * np.prod(self.shape)), dtype=self.dtype), \
                np.zeros((self.train_val_nsamples,), dtype=self.dtype)
            self.x_test, self.y_test = \
                np.empty((self.test_nsamples * np.prod(self.shape)), dtype=self.dtype), \
                np.zeros((self.test_nsamples,), dtype=self.dtype)
        else:
            x_train_fname = "train-images-idx3-ubyte"
            y_train_fname = "train-labels-idx1-ubyte"
            x_test_fname = "t10k-images-idx3-ubyte"
            y_test_fname = "t10k-labels-idx1-ubyte"

            self.x_train_val = self.__read_file("%s/%s" % (self.train_path, x_train_fname))
            self.y_train_val = self.__read_file("%s/%s" % (self.train_path, y_train_fname))
            self.x_test = self.__read_file("%s/%s" % (self.test_path, x_test_fname))
            self.y_test = self.__read_file("%s/%s" % (self.test_path, y_test_fname))

        self.x_train_val = self.x_train_val.flatten().reshape(self.train_val_nsamples, *self.shape).astype(
            self.dtype) / 255.0
        self.y_train_val = self.__one_hot_encoder(self.y_train_val.astype(np.int16))
        self.x_test = self.x_test.flatten().reshape(self.test_nsamples, *self.shape).astype(self.dtype) / 255.0
        self.y_test = self.__one_hot_encoder(self.y_test.astype(np.int16))

        if self.test_as_validation:
            # print("  Using test as validation data - val_split parameter is ignored!")
            self.x_val, self.y_val = self.x_test, self.y_test
            self.x_train, self.y_train = self.x_train_val, self.y_train_val
            self.train_nsamples = self.x_train.shape[0]

        # Attributes defined later
        self.val_size = None

    @staticmethod
    def __read_file(fname):
        with open(fname, 'rb') as f:
            zero, data_type, dims = struct.unpack('>HBB', f.read(4))
            shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(dims))
            return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

    def __one_hot_encoder(self, y):
        y_one_hot = np.zeros((y.shape[0], self.nclasses), dtype=self.dtype, order="C")
        y_one_hot[np.arange(y.shape[0]), y] = 1
        return y_one_hot

    def make_train_val_partitions(self, val_split=0.2):
        if self.test_as_validation:
            return
        assert 0 <= val_split < 1
        self.val_size = int(self.train_val_nsamples * val_split)

        end = self.val_start + self.val_size
        if end > self.x_train_val.shape[0]:
            val_indices = np.arange(self.val_start, self.x_train_val.shape[0])
            self.val_start = self.val_size - val_indices.shape[0]
            val_indices = np.concatenate((val_indices, np.arange(0, self.val_start)))
        else:
            val_indices = np.arange(self.val_start, end)
            self.val_start = end

        train_indices = np.setdiff1d(np.arange(self.train_val_nsamples), val_indices)

        self.x_train = self.x_train_val[train_indices, ...]
        self.y_train = self.y_train_val[train_indices, ...]
        self.x_val = self.x_train_val[val_indices, ...]
        self.y_val = self.y_train_val[val_indices, ...]
        self.train_nsamples = self.x_train.shape[0]

    def adjust_steps_per_epoch(self, steps_per_epoch, local_batch_size, nprocs):
        if steps_per_epoch > 0:
            subset_size = local_batch_size * nprocs * steps_per_epoch
            if subset_size > self.x_train_val.shape[0]:
                scale = math.ceil(subset_size / float(self.x_train_val.shape[0]))
                self.x_train_val = np.tile(self.x_train_val, scale)[:subset_size, ...]
                self.y_train_val = np.tile(self.y_train_val, scale)[:subset_size, ...]
            else:
                self.x_train_val = self.x_train_val[:subset_size, ...]
                self.y_train_val = self.y_train_val[:subset_size, ...]
            self.train_val_nsamples = self.x_train_val.shape[0]
            self.train_nsamples = self.train_val_nsamples
            if self.test_as_validation:
                self.x_train, self.y_train = self.x_train_val, self.y_train_val


class CIFAR10(Dataset):

    def __init__(self, train_path, test_path, model="", test_as_validation=False,
                 flip_images=False, flip_images_prob=0.5,
                 crop_images=False, crop_images_size=16, crop_images_prob=0.5,
                 dtype=np.float32, use_synthetic_data=False):
        self.train_path = train_path
        self.test_path = test_path
        self.model = model
        self.test_as_validation = test_as_validation
        self.flip_images = flip_images
        self.flip_images_prob = flip_images_prob
        self.crop_images = crop_images
        self.crop_images_size = crop_images_size
        self.crop_images_prob = crop_images_prob
        self.dtype = dtype
        self.use_synthetic_data = use_synthetic_data
        self.nclasses = 10
        self.val_start = 0

        self.train_val_nsamples = 50000
        self.test_nsamples = 10000

        self.images_per_file = 10000
        self.shape = (3, 32, 32)

        xy_train_fname = "data_batch_%d.bin"
        xy_test_fname = "test_batch.bin"

        for b in range(1, 6):
            if self.use_synthetic_data:
                self.x_train_val_aux, self.y_train_val_aux = \
                    np.empty((self.images_per_file * np.prod(self.shape)), dtype=self.dtype), \
                    np.zeros((self.images_per_file,), dtype=self.dtype)
            else:
                self.x_train_val_aux, self.y_train_val_aux = \
                    self.__read_file("%s/%s" % (self.train_path, (xy_train_fname % b)))
            if b == 1:
                self.x_train_val, self.y_train_val = self.x_train_val_aux, self.y_train_val_aux
            else:
                self.x_train_val = np.concatenate((self.x_train_val, self.x_train_val_aux), axis=0)
                self.y_train_val = np.concatenate((self.y_train_val, self.y_train_val_aux), axis=0)

        if self.use_synthetic_data:
            self.x_test, self.y_test = \
                np.empty((self.images_per_file * np.prod(self.shape)), dtype=self.dtype), \
                np.zeros((self.images_per_file,), dtype=self.dtype)
        else:
            self.x_test, self.y_test = self.__read_file("%s/%s" % (self.test_path, xy_test_fname))

        self.x_train_val = self.x_train_val.reshape(self.train_val_nsamples, *self.shape).astype(self.dtype) / 255.0
        # self.x_train_val = self.__normalize_image(self.x_train_val)
        self.y_train_val = self.__one_hot_encoder(self.y_train_val.astype(np.int16))
        self.x_test = self.x_test.reshape(self.test_nsamples, *self.shape).astype(self.dtype) / 255.0
        # self.x_test = self.__normalize_image(self.x_test)
        self.y_test = self.__one_hot_encoder(self.y_test.astype(np.int16))

        if self.test_as_validation:
            # print("  Using test as validation data - val_split parameter is ignored!")
            self.x_val, self.y_val = self.x_test, self.y_test
            self.x_train, self.y_train = self.x_train_val, self.y_train_val
            self.train_nsamples = self.x_train.shape[0]

    def __read_file(self, fname):
        with open(fname, 'rb') as f:
            im = np.frombuffer(f.read(), dtype=np.uint8).reshape(self.images_per_file, np.prod(self.shape) + 1)
            y, x = im[:, 0].flatten(), im[:, 1:].flatten()
            return x, y

    @staticmethod
    def __normalize_image(x):
        mean = np.mean(x, axis=(0, 2, 3))
        std = np.std(x, axis=(0, 2, 3))
        for c in range(3):
            x[:, c, ...] = (x[:, c, ...] - mean[c]) / std[c]
        return x

    def __one_hot_encoder(self, y):
        y_one_hot = np.zeros((y.shape[0], self.nclasses), dtype=self.dtype, order="C")
        y_one_hot[np.arange(y.shape[0]), y] = 1
        return y_one_hot

    def make_train_val_partitions(self, val_split=0.2):
        if self.test_as_validation:
            return
        assert 0 <= val_split < 1
        val_size = int(self.train_val_nsamples * val_split)

        end = self.val_start + val_size
        if end > self.x_train_val.shape[0]:
            val_indices = np.arange(self.val_start, self.x_train_val.shape[0])
            self.val_start = val_size - val_indices.shape[0]
            val_indices = np.concatenate((val_indices, np.arange(0, self.val_start)))
        else:
            val_indices = np.arange(self.val_start, end)
            self.val_start = end

        train_indices = np.setdiff1d(np.arange(self.train_val_nsamples), val_indices)

        self.x_train = self.x_train_val[train_indices, ...]
        self.y_train = self.y_train_val[train_indices, ...]
        self.x_val = self.x_train_val[val_indices, ...]
        self.y_val = self.y_train_val[val_indices, ...]
        self.train_nsamples = self.x_train.shape[0]

    def adjust_steps_per_epoch(self, steps_per_epoch, local_batch_size, nprocs):
        if steps_per_epoch > 0:
            subset_size = local_batch_size * nprocs * steps_per_epoch
            if subset_size > self.x_train_val.shape[0]:
                scale = math.ceil(subset_size / float(self.x_train_val.shape[0]))
                self.x_train_val = np.tile(self.x_train_val, scale)[:subset_size, ...]
                self.y_train_val = np.tile(self.y_train_val, scale)[:subset_size, ...]
            else:
                self.x_train_val = self.x_train_val[:subset_size, ...]
                self.y_train_val = self.y_train_val[:subset_size, ...]
            self.train_val_nsamples = self.x_train_val.shape[0]
            self.train_nsamples = self.train_val_nsamples
            if self.test_as_validation:
                self.x_train, self.y_train = self.x_train_val, self.y_train_val


class ImageNet(Dataset):

    def __init__(self, train_path, test_path, model="", test_as_validation=False,
                 flip_images=False, flip_images_prob=0.5,
                 crop_images=False, crop_images_size=112, crop_images_prob=0.5,
                 dtype=np.float32, use_synthetic_data=False):
        self.train_path = self.val_path = train_path
        self.test_path = test_path
        self.model = model
        self.flip_images = flip_images
        self.flip_images_prob = flip_images_prob
        self.crop_images = crop_images
        self.crop_images_size = crop_images_size
        self.crop_images_prob = crop_images_prob
        self.test_as_validation = test_as_validation
        self.dtype = dtype
        self.use_synthetic_data = use_synthetic_data
        self.nclasses = 1000
        self.val_start = 0

        self.images_per_train_file, self.images_per_test_file = 1251, 390
        self.shape = (3, 227, 227)

        # Variables for training + validation datasets
        if self.use_synthetic_data:
            self.n_train_val_files = 1024
            self.train_val_files = [''] * self.n_train_val_files
        else:
            self.train_val_files = os.listdir(self.train_path)
            self.train_val_files.sort()
            self.n_train_val_files = len(self.train_val_files)

        self.train_val_nsamples = self.n_train_val_files * self.images_per_train_file

        # Variables for testing dataset
        if self.use_synthetic_data:
            self.n_test_files = 1  # 128
            self.test_files = [''] * self.n_test_files
        else:
            self.test_files = os.listdir(self.test_path)
            self.test_files.sort()
            self.n_test_files = len(self.test_files)

        self.test_nsamples = self.n_test_files * self.images_per_test_file

        if self.test_as_validation:
            # print("  Using test as validation data - val_split parameter is ignored!")
            self.val_path, self.val_files = self.test_path, self.test_files
            self.train_files = self.train_val_files
            self.train_nsamples = self.train_val_nsamples

        # Attributes defined later
        self.val_size = None

    def __normalize_image(self, x):
        if "alexnet" not in self.model:  # for VGG, ResNet and other models input shape must be (3,224,224)
            return x[..., 1:225, 1:225]
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        for c in range(3):
            x[:, c, ...] = ((x[:, c, ...] / 255.0) - mean[c]) / std[c]
        return x

    def __one_hot_encoder(self, y):
        y_one_hot = np.zeros((y.shape[0], self.nclasses), dtype=self.dtype, order="C")
        y_one_hot[np.arange(y.shape[0]), y] = 1
        return y_one_hot

    def data_generator(self, path, files, batch_size, op="train"):
        # For batch sizes > 1251 it is needed to concatenate more than one file of 1251 samples
        # In this case we yield bigger chunks of size batch_size
        # The next variable is not used
        # images_per_file = {"train": self.images_per_train_file,
        #                    "test": self.images_per_test_file}[op]
        in_files = files.copy()
        np.random.shuffle(in_files)

        if batch_size > self.images_per_train_file:
            x_buffer, y_buffer = np.array([]), np.array([])

            for f in in_files:
                if self.use_synthetic_data:
                    images = {"train": self.images_per_train_file,
                              "test": self.images_per_test_file}[op]
                    values = {"x": np.empty((images, *self.shape), dtype=self.dtype),
                              "y": np.zeros((images, 1), dtype=self.dtype)}
                else:
                    values = np.load("%s/%s" % (path, f))

                x_data = self.__normalize_image(values['x'].astype(self.dtype))
                y_data = self.__one_hot_encoder(values['y'].astype(np.int16).flatten() - 1)
                if x_buffer.size == 0:
                    x_buffer, y_buffer = x_data, y_data
                else:
                    x_buffer = np.concatenate((x_buffer, x_data), axis=0)
                    y_buffer = np.concatenate((y_buffer, y_data), axis=0)

                if x_buffer.shape[0] >= batch_size:
                    if self.flip_images:
                        x_buffer = do_flip_images(x_buffer, self.flip_images_prob)
                    if self.crop_images:
                        x_buffer = do_crop_images(x_buffer, self.crop_images_size, self.crop_images_prob)
                    yield x_buffer[:batch_size, ...], y_buffer[:batch_size, ...]
                    x_buffer = x_buffer[batch_size:, ...]
                    y_buffer = y_buffer[batch_size:, ...]
                    gc.collect()

            if x_buffer.shape[0] > 0:
                yield x_buffer, y_buffer
                gc.collect()

        # For batch_sizes <= 1251, complete files of 1251 samples are yield
        else:
            for f in in_files:
                if self.use_synthetic_data:
                    images = {"train": self.images_per_train_file,
                              "test": self.images_per_test_file}[op]
                    values = {"x": np.empty((images, *self.shape), dtype=self.dtype),
                              "y": np.zeros((images, 1), dtype=self.dtype)}
                else:
                    values = np.load("%s/%s" % (path, f))

                x_data = self.__normalize_image(values['x'].astype(self.dtype))
                y_data = self.__one_hot_encoder(values['y'].astype(np.int16).flatten() - 1)
                if self.flip_images:
                    x_data = do_flip_images(x_data, self.flip_images_prob)
                if self.crop_images:
                    x_data = do_crop_images(x_data, self.crop_images_size, self.crop_images_prob)
                yield x_data, y_data
                gc.collect()

    def train_data_generator(self, batch_size):
        return self.data_generator(self.train_path, self.train_files, batch_size, op="train")

    def val_data_generator(self, batch_size):
        return self.data_generator(self.val_path, self.val_files, batch_size, op="train")

    def test_data_generator(self, batch_size):
        return self.data_generator(self.test_path, self.test_files, batch_size, op="test")

    def make_train_val_partitions(self, val_split=0.2):
        if self.test_as_validation:
            return
        assert 0 <= val_split < 1
        self.val_size = int((self.train_val_nsamples * val_split) / self.images_per_train_file)

        end = self.val_start + self.val_size
        if end > self.n_train_val_files:
            val_idx_files = np.arange(self.val_start, self.n_train_val_files)
            self.val_start = self.val_size - val_idx_files.shape[0]
            val_idx_files = np.concatenate((val_idx_files, np.arange(0, self.val_start)))
        else:
            val_idx_files = np.arange(self.val_start, end)
            self.val_start = end

        train_idx_files = np.setdiff1d(np.arange(self.n_train_val_files), val_idx_files)
        self.train_files = [self.train_val_files[f] for f in train_idx_files]
        self.val_files = [self.train_val_files[f] for f in val_idx_files]
        self.train_nsamples = len(self.train_files) * self.images_per_train_file

    def adjust_steps_per_epoch(self, steps_per_epoch, local_batch_size, nprocs):
        if steps_per_epoch > 0:
            subset_size = local_batch_size * nprocs * steps_per_epoch
            if subset_size < self.train_val_nsamples:
                subset_files = max(1, subset_size // self.images_per_train_file)
                self.train_val_files = self.train_val_files[:subset_files]
                self.n_train_val_files = len(self.train_val_files)
                self.train_val_nsamples = self.n_train_val_files * self.images_per_train_file
                self.train_nsamples = self.train_val_nsamples
                subset_test_files = max(1, subset_size // self.images_per_test_file)
                self.test_files = self.test_files[:subset_test_files]
                self.n_test_files = len(self.test_files)
                self.test_nsamples = self.n_test_files * self.images_per_test_file
