""" Python Distributed Training of Neural Networks - PyDTNN

PyDTNN is a light-weight library for distributed Deep Learning training and 
inference that offers an initial starting point for interaction with 
distributed training of (and inference with) deep neural networks. PyDTNN 
priorizes simplicity over efficiency, providing an amiable user interface 
which enables a flat accessing curve. To perform the training and inference 
processes, PyDTNN exploits distributed inter-process parallelism (via MPI) 
for clusters and intra-process (via multi-threading) parallelism to leverage 
the presence of multicore processors and GPUs at node level. For that, PyDTNN 
uses MPI4Py for message-passing, BLAS calls via NumPy for multicore processors
and PyCUDA+cuDNN+cuBLAS for NVIDIA GPUs.

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
__version__ = "1.1.0"


import ctypes, os
from importlib import import_module

PYDL_EVT = 60000001
PYDL_OPS_EVT = 60000002
PYDL_NUM_EVTS = 5
PYDL_OPS_NUM_EVTS = 6

class Tracer:

    def __init__(self, tracing=False):        
        self.tracing = tracing
        if self.tracing:
            self.pyextrae = import_module('pyextrae.common.extrae')

    def define_event_type(self, model):
        if self.tracing:
            nvalues = len(model.layers) * PYDL_NUM_EVTS + 1
            description = "Model layers"
            values = (ctypes.c_ulonglong * nvalues)()
            description_values = (ctypes.c_char_p * nvalues)()
            values[0] = 0
            description_values[0] = "End".encode('utf-8')
            for i in range(1, nvalues):
              values[i] = i
            for i in range(len(model.layers)):
              description_values[i*PYDL_NUM_EVTS+1] = (str(i) + "_" + type(model.layers[i]).__name__ + "_forward ").encode('utf-8')
              description_values[i*PYDL_NUM_EVTS+2] = (str(i) + "_" + type(model.layers[i]).__name__ + "_backward ").encode('utf-8')
              description_values[i*PYDL_NUM_EVTS+3] = (str(i) + "_" + type(model.layers[i]).__name__ + "_allreduce_dw ").encode('utf-8')
              description_values[i*PYDL_NUM_EVTS+4] = (str(i) + "_" + type(model.layers[i]).__name__ + "_wait_dw ").encode('utf-8')
              description_values[i*PYDL_NUM_EVTS+5] = (str(i) + "_" + type(model.layers[i]).__name__ + "_update_dw ").encode('utf-8')
    
            self.pyextrae.Extrae[os.getpid()].Extrae_define_event_type(
                ctypes.pointer(ctypes.c_uint(PYDL_EVT)),
                ctypes.c_char_p(description.encode('utf-8')),
                ctypes.pointer(ctypes.c_uint(nvalues)),
                ctypes.pointer(values),
                ctypes.pointer(description_values) )
    
            nvalues = len(model.layers) * PYDL_OPS_NUM_EVTS + 1
            description = "PYDL ops per layer"
            values = (ctypes.c_ulonglong * nvalues)()
            description_values = (ctypes.c_char_p * nvalues)()
            values[0] = 0
            description_values[0] = "End".encode('utf-8')
            for i in range(1, nvalues):
              values[i] = i
            for i in range(len(model.layers)):
              description_values[i*PYDL_OPS_NUM_EVTS+1] = (str(i) + "_" + type(model.layers[i]).__name__ + "_forward_matmul ").encode('utf-8')
              description_values[i*PYDL_OPS_NUM_EVTS+2] = (str(i) + "_" + type(model.layers[i]).__name__ + "_forward_im2col ").encode('utf-8')
              description_values[i*PYDL_OPS_NUM_EVTS+3] = (str(i) + "_" + type(model.layers[i]).__name__ + "_compute_dx_matmul ").encode('utf-8')
              description_values[i*PYDL_OPS_NUM_EVTS+4] = (str(i) + "_" + type(model.layers[i]).__name__ + "_compute_dx_col2im ").encode('utf-8')
              description_values[i*PYDL_OPS_NUM_EVTS+5] = (str(i) + "_" + type(model.layers[i]).__name__ + "_compute_dw_matmul ").encode('utf-8')
              description_values[i*PYDL_OPS_NUM_EVTS+6] = (str(i) + "_" + type(model.layers[i]).__name__ + "_allreduce_dw ").encode('utf-8')
    
            self.pyextrae.Extrae[os.getpid()].Extrae_define_event_type(
                ctypes.pointer(ctypes.c_uint(PYDL_OPS_EVT)),
                ctypes.c_char_p(description.encode('utf-8')),
                ctypes.pointer(ctypes.c_uint(nvalues)),
                ctypes.pointer(values),
                ctypes.pointer(description_values) )
    
    def emit_event(self, evt, val):
        if self.tracing:
            self.pyextrae.eventandcounters(evt, val)
 
    def emit_nevent(self, evt, val):
        if self.tracing:
            self.pyextrae.neventandcounters(evt, val)
 
