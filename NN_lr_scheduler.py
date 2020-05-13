""" Python Distributed Training of Neural Networks - PyDTNN

PyDTNN is a light-weight library for distributed Deep Learning training and 
inference that offers an initial starting point for interaction with 
distributed training of (and inference with) deep neural networks. PyDTNN 
priorizes simplicity over efficiency, providing an amiable user interface 
which enables a flat accessing curve. To perform the training and inference 
processes, PyDTNN exploits distributed inter-process parallelism (via MPI) 
for clusters and intra-process (via multi-threading) parallelism to leverage 
the presence of multicore processors at node level.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.

"""

__author__ = "Manuel F. Dolz, Enrique S. Quintana, \
              Mar Catalan, Adrian Castello"
__contact__ = "dolzm@uji.es"
__copyright__ = "Copyright 2020, Universitat Jaume I"
__credits__ = ["Manuel F. Dolz, Enrique S. Quintana", \
               "Mar Catalan", "Adrian Castello"]
__date__ = "2020/03/22"

__email__ =  "dolzm@uji.es"
__license__ = "GPLv3"
__maintainer__ = "Manuel F. Dolz"
__status__ = "Production"
__version__ = "1.0.1"


import time, os
import numpy as np
from NN_layer import Layer

class LRScheduler():

    def __init__(self):
        pass

    def on_batch_begin(self, *args):
        pass

    def on_batch_end(self, *args):
        pass

    def on_epoch_begin(self, *args):
        pass

    def on_epoch_end(self, *args):
        pass


class WarmUpLRScheduler(LRScheduler):

    def __init__(self, warmup_batches=100, init_lr=1e-3, verbose=True):
        super().__init__()
        self.warmup_batches = warmup_batches
        self.init_lr = init_lr
        self.verbose = verbose
        self.batch_count = 0

    def on_batch_begin(self, model, optimizer, rank):
        if self.batch_count <= self.warmup_batches:
            optimizer.learning_rate = \
                self.batch_count * self.init_lr / self.warmup_batches
            self.batch_count += 1
            if self.verbose and rank == 0:
                print("LRScheduler %s: setting learning rate to %.8f" % \
                    (type(self).__name__, optimizer.learning_rate))


class EarlyStopping(LRScheduler):

    def __init__(self, loss_metric="", patience=10, verbose=True):
        super().__init__()
        self.loss_metric = loss_metric
        self.is_val_metric = "val_" in self.loss_metric
        check_val = self.loss_metric.split("_")
        if "val" == check_val[0]:
            self.loss_metric_ = "_".join(check_val[1:])
        self.patience = patience
        self.epoch_count = self.best_epoch = 0
        self.stop_training = False
        self.best_loss = np.inf * {True: -1, False: 1}["accuracy" in self.loss_metric]
        self.best_weights_filename = None
        self.verbose = verbose

    def on_epoch_end(self, model, optimizer, loss_metrics, train_loss, val_loss, rank):
        try:    idx = loss_metrics.index(self.loss_metric_)
        except: idx = 0
        self.epoch_count += 1

        loss = {True: val_loss, False: train_loss}[self.is_val_metric]
        if ("accuracy" in self.loss_metric and loss[idx] > self.best_loss) or \
           ("accuracy" not in self.loss_metric and loss[idx] < self.best_loss):
            self.best_loss = loss[idx]
            self.best_epoch = self.epoch_count
            # Save weights + bias
            if not self.best_weights_filename: 
                self.best_weights_filename = "./model-%s-weights-%s.npz" % \
                    (model.params.model, time.strftime("%Y%m%d"))   
            model.store_weights_and_bias(self.best_weights_filename)
        elif (self.epoch_count - self.best_epoch) >= self.patience:
            self.stop_training = True
            # Restore weights + bias
            model.load_weights_and_bias(self.best_weights_filename)
            if self.verbose and rank == 0:
                print("LRScheduler %s: metric %s did not improve for %d epochs, stop training!" % \
                     (type(self).__name__, self.loss_metric, self.patience))


class ReduceLROnPlateau(LRScheduler):

    def __init__(self, loss_metric="", factor=0.1, patience=5, min_lr=0, verbose=True):
        super().__init__()
        self.loss_metric = loss_metric
        self.is_val_metric = "val_" in self.loss_metric
        check_val = self.loss_metric.split("_")
        if "val" == check_val[0]:
            self.loss_metric_ = "_".join(check_val[1:])
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.epoch_count = self.best_epoch = 0
        self.best_loss = np.inf * {True: -1, False: 1}["accuracy" in self.loss_metric]
        self.verbose = verbose

    def on_epoch_end(self, model, optimizer, loss_metrics, train_loss, val_loss, rank):
        try:    idx = loss_metrics.index(self.loss_metric_)
        except: idx = 0
        self.epoch_count += 1
        
        loss = {True: val_loss, False: train_loss}[self.is_val_metric]
        if ("accuracy" in self.loss_metric and loss[idx] > self.best_loss) or \
           ("accuracy" not in self.loss_metric and loss[idx] < self.best_loss):
            self.best_loss = loss[idx]
            self.best_epoch = self.epoch_count
        elif (self.epoch_count - self.best_epoch) >= self.patience and \
             (optimizer.learning_rate * self.factor) >= self.min_lr:
            optimizer.learning_rate *= self.factor
            self.best_epoch = self.epoch_count
            if self.verbose and rank == 0:
                print("LRScheduler %s: metric %s did not improve for %d epochs, setting learning rate to %.8f" % \
                     (type(self).__name__, self.loss_metric, 
                        self.patience, optimizer.learning_rate))


class ModelCheckpoint(LRScheduler):

    def __init__(self, loss_metric="", epoch_save_frequency=1, verbose=True):
        super().__init__()
        self.loss_metric = loss_metric
        self.is_val_metric = "val_" in self.loss_metric
        check_val = self.loss_metric.split("_")
        if "val" == check_val[0]:
            self.loss_metric_ = "_".join(check_val[1:])
        self.epoch_save_frequency = epoch_save_frequency
        self.epoch_count = self.best_epoch = 0
        self.best_loss = np.inf * {True: -1, False: 1}["accuracy" in self.loss_metric]
        self.last_filename = ""
        self.verbose = verbose

    def on_epoch_end(self, model, optimizer, loss_metrics, train_loss, val_loss, rank):
        try:    idx = loss_metrics.index(self.loss_metric)
        except: idx = 0
        self.epoch_count += 1
        d = {}

        loss = {True: val_loss, False: train_loss}[self.is_val_metric]
        if ("accuracy" in self.loss_metric and loss[idx] > self.best_loss) or \
           ("accuracy" not in self.loss_metric and loss[idx] < self.best_loss):
            self.best_loss = loss[idx]
            self.best_epoch = self.epoch_count
            if self.epoch_count % self.epoch_save_frequency == 0:           
                self.filename="./model-%s-epoch-%d-%s.npz" % \
                    (model.params.model, self.epoch_count, time.strftime("%Y%m%d"))
                model.store_weights_and_bias(self.filename)
                if self.verbose and rank == 0:
                    print("LRScheduler %s: saving model weights and bias in %s" % \
                         (type(self).__name__, self.filename))
                if self.last_filename: os.remove(self.last_filename)
                self.last_filename = self.filename

            
