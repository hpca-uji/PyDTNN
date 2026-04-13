# Contributing guidelines
## Imports
- Do not re-export symbols (ie from `__init__.py`), always import from the original module.
- Avoid star imports (`from x import *`), they break type checking and IDE support.
- Avoid defining `__init__.py` unless strictly necessary (it can easily introduce circular imports).

## Numpy
- Prefer `numpy` functions over operators when possible (they may offer better numerical precision).
- Do not create arrays using `np.ndarray()`, use explicit initializers such as `np.zeros()`, `np.ones()`, `np.arange()`, or `np.empty()`.
- Avoid magic numbers in `np.transpose(ary, format)`, use `format_transpose(ary, src, dst)` instead.

## Cython
- Cython's `.pyx` can be included anywhere, but must be accompanied by a `.pyi` typing interface.

## Random
- Use `pydtnn.utils.random` for random number generation. Other generators are not multi-thread aware.

## GPU
- Each CUDA kernel must have a unique name. Duplicate names will cause incorrect kernel resolution.

## Tests
- Test all changes across backends (CPU, GPU, etc.), changes in base classes may introduce backend-specific issues.
- When comparing outputs between layers or models, always copy outputs before passing them to the next layer, some layers perform in-place operations.

## Configuration
- Keep `README.md` and `parser.py` in sync, any change in options must be reflected in both.



# Architecture notes
- In components, `__init__` is used for model-agnostic configuration, and `_model_init` for model specific configuration and resource allocation.

## GPU
- `--enable-gpudirect` moves data from CPU (`ndarray`) to GPU (`GPUArray`), requires `enable-cudnn`
- `--enable-nccl` moves reductions from CPU (`MPI`) to GPU (`NCCL`), requires `enable-gpudirect`

## Encryption
- Requires `NCCL` to be disabled (otherwise it will be skipped), typically requires
  `--use-mpi-buffers=False` (crypto libraries usually do not expose buffer access) and
  `--use-blocking-mpi=True` (MPI like `mpi4py` does not support async object reductions)

## Memory
- When using `PreallocMemory`, temporary memory in block layers will overlap its child layers, therefore it may be overwritten.

## Troubleshoot
- If using `conda` and `pip install` fails with `no such option: --config-settings`, deactivate all environments and reactivate only the target environment.



# Planned changes
- Replace `cupy-cuda` with `cupy` (for AMD ROCm support)
- Move `gpu.utils.memory_allocation` from global scope to model instance
- Extract shared logic from `.pyi` and `.pyx` into a common module
- Move GPU `SourceModule` code into `.cu` files



# Publishing guide
Dependencies: `gcc patchelf` and `build twine auditwheel`

```sh
python -m build --outdir ./dist/
python -m build --outdir ./build/ --wheel
python -m auditwheel repair --wheel-dir ./dist/ ./build/*.whl
python -m twine upload --repository pypi ./dist/*
```



# Project structure
## Repository root
```
├── README.md
├── CONTRIBUTING.md
├── pyproject.toml
├── setup.py
├── LICENSE
├── # other resources
```
## Python package
```
├── pydtnn
│   ├── logging.yaml
│   ├── pydtnn_benchmark.py
│   ├── parser.py
│   ├── model.py
│   ├── layer_base.py
```
### Components
```
│   ├── activations
│   │   ├── activation.py  # base
│   │   └── # each implementation
│   ├── models
│   │   ├── # each implementation
│   ├── layers
│   │   ├── abstract  # shared
│   │   ├── layer.py  # base
│   │   └── # each implementation
│   ├── losses
│   │   ├── loss.py  # base
│   │   └── # each implementation
│   ├── metrics
│   │   ├── metric.py  # base
│   │   └── # each implementation
│   ├── schedulers
│   │   ├── scheduler.py  # base
│   │   └── # each implementation
│   ├── optimizers
│   │   ├── optimizer.py  # base
│   │   └── # each implementation
│   ├── backends
│   │   ├── __init__.py  # base
│   │   ├── numpy 
│   │   ├── cython
│   │   │   ├── # implementation
│   │   │   └── utils  # pyx & pyi
│   │   ├── # each implementation with whole components structure
│   │   └── pycuda
│   │       ├── # implentation
│   │       └── utils
│   │           ├── memory_allocation.py
│   │           └── tensor_array.py
```
### Support modules
```
│   ├── datasets
│   │   ├── dataset.py  # base
│   │   ├── archive.py
│   │   ├── memory.py
│   │   ├── folder.py
│   │   ├── synthetic.py
│   │   └── # each implementation
│   ├── tracers
│   │   ├── events.py
│   │   ├── tracer.py
│   │   └── # each implementation
│   ├── tests
│   │   ├── README.md
│   │   ├── groups  # test groupings
│   │   └── abstract  # base test cases
│   ├── converters
│   │   ├── README.md
│   │   ├── onnx2pydtnn
│   │   ├── pydtnn2onnx
│   │   └── pytorch2pydtnn
│   ├── libs
│   │   ├── # bindings to libraries
│   │   └── utils.py
│   └── utils
│       ├── constants.py
│       ├── initializers.py
│       ├── debug.py
│       ├── gpu.py
│       ├── memory_pool.py
│       ├── pmlib.py
│       ├── profiler.py
│       ├── random.py
│       ├── tensor.py
│       └── # other utilities
```
## Support files
```
├── scripts
│   ├── README.md
│   ├── models
│   ├── datasets
│   ├── extrae
│   ├── profilers
│   ├── tests
│   └── utils
├── vendor
│   ├── README.md
│   └── # each repository
└── datasets
    └── # each dataset
```