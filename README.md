# Python Distributed Training of Neural Networks - PyDTNN

## Introduction ##

**PyDTNN** is a light-weight library for distributed Deep Learning training 
and inference that offers an initial starting point for interaction 
with distributed training of (and inference with) deep neural networks. 
PyDTNN priorizes simplicity over efficiency, providing an amiable user 
interface which enables a flat accessing curve. To perform the training and 
inference processes, PyDTNN exploits distributed inter-process parallelism 
(via MPI) for clusters and intra-process (via multi-threading) parallelism 
to leverage the presence of multicore processors at node level. For that, 
PyDTNN uses MPI4Py for message-passing and im2col transforms to cast the 
convolutions in terms of dense matrix-matrix multiplications, 
which are realized BLAS calls via NumPy.

Currently, **PyDTNN** supports the followint layers:

  * Fully-connected
  * Convolutional
  * Max pooling
  * Average pooling

[comment]: ## Install and compile instructions

[comment]: See the [install and compile notes](doc/install-notes.md).

[comment]: ## Publications describing PyDTNN

[comment]: ### References

The **PyDTNN** library has been partially supported by:

* Project TIN2017-82972-R **"Agorithmic Techniques for Energy-Aware and Error-Resilient High Performance Computing"** funded by the Spanish Ministry of Economy and Competitiveness (2018-2020).

* Project RTI2018-098156-B-C51 **"Innovative Technologies of Processors, Accelerators and Networks for Data Centers and High Performance Computing"** funded by the Spanish Ministry of Science, Innocation and Universities .

* Project CDEIGENT/2017/04 **"High Performance Computing for Neural Networks"** funded by the Valencian Government.
