# Guidelines
_Things to do & not to do_

## Exports
- Do not re-export symbols (ie from `__init__.py`), always import from the original module.
- Avoid star imports (`from x import *`), they break type checking and IDE support.
- Avoid defining `__init__.py` unless strictly necessary (it can easily introduce circular imports).

## NumPy
- Prefer `numpy` functions over operators when possible (they may offer better numerical precision).
- Do not create arrays using `np.ndarray()`, use explicit initializers such as `np.zeros()`, `np.ones()`, `np.arange()`, or `np.empty()`.
- Avoid magic numbers in `np.transpose(ary, format)`, use `format_transpose(ary, src, dst)` instead.

## Cython
- Cython's `.pyx` can be included anywhere, but must be accompanied by a `.pyi` typing interface.

## CUDA
- Each CUDA kernel must have a unique name, duplicate names will cause incorrect kernel resolution.

## Randomness
- Use `pydtnn.utils.random` (or a local instance) for random number generation, other generators are not multi-thread aware.

## Tests
- Test all changes across backends (CPU, GPU, etc.), changes in base classes may introduce backend-specific issues.
- When comparing outputs between layers or models, always copy outputs before passing them to the next layer, some layers perform in-place operations.

## Documentation
- Keep `README.md`, `utils/parser.py` and `model/base.py` in sync, any change in options must be reflected in all of them.

## Environment
- Use `from __future__ import annotations` instead of string type annotations, it is more legible and will be default moving forward.
- Keep `Makefile` and `pyproject.toml` in sync, any changes in dependencies versions must be reflected in both.

## Components
- If components structure changes, update the `Structure` section of this document accordingly.
- If the Model or Dataset components structure changes, update the class's `__init__` diagram.

# Knowledge
_Things you should keep in mind_

## Bootstrapping
- In components, `__init__` is used for model-agnostic configuration, and `_model_init` for model specific configuration, and `_post_init` and resource allocation.

## GPU
- `--use-gpudirect` moves data from CPU (`ndarray`) to GPU (`GPUArray`), requires `enable-cudnn`
- `--use-nccl` moves reductions from CPU (`MPI`) to GPU (`NCCL`), requires `enable-gpudirect`

## FHE
- `--enable-encryption` requires `NCCL` to be disabled (otherwise it will be skipped),
  typically requires `--use-mpi-buffers=False` (crypto libraries usually do not expose buffer access)
  and `--use-blocking-mpi=True` (MPI like `mpi4py` does not support async object reductions)

## Cython
- Shared interface and typing code can be defined in `.pyd` files.
- Multiple Cython optimizations are enabled by default, check for them in `setup.py`, and if desired disabled them locally with `@cython.{option}(value)`.

## Memory
- When using `PreallocMemory`, temporary memory in block layers will overlap its child layers, therefore it may be overwritten.

## Environment
- If using `conda` and `pip install` fails with `no such option: --config-settings`, deactivate all environments and reactivate only the target environment.

## Typing
- If sure, you can disable typing errors with `# typing: ignore (reason)`.
- If sure, you can disable styling errors with `# noqa: {error} (reason)`.

# Structure
_How is the project organized_

## Repository root
```
├── README.md
├── CONTRIBUTING.md
├── Makefile
├── pyproject.toml
├── setup.py
├── .editor
├── .mailmap
├── LICENSE
├── # other resources
```
## Python package
```
├── pydtnn
│   ├── __main__.py
│   ├── __init__.py
│   ├── logging.yaml
```
### Components
```
│   ├── model
│   │   ├── __init__.py  # usable models
│   │   ├── base.py      # typing interface
│   │   ├── utils.py     # utility methods
│   │   ├── layers.py    # layers management
│   │   ├── state.py     # state management
│   │   ├── init.py      # initialization
│   │   ├── sync.py      # synchronization
│   │   ├── repr.py      # representation
│   │   ├── eval.py      # model inference
│   │   └── train.py     # model training
|   ├── abstract
|   |   ├── base.py       # every component
│   │   └── layerable.py  # layer-like component
│   ├── activations
│   │   ├── activation.py  # base
│   │   └── # each implementation
│   ├── models
│   │   ├── # each description
│   ├── layers
│   │   ├── abstract      # shared
│   │   ├── layer.py      # base
│   │   └── # each implementation
│   ├── losses
│   │   ├── loss.py       # base
│   │   └── # each implementation
│   ├── metrics
│   │   ├── metric.py     # base
│   │   └── # each implementation
│   ├── schedulers
│   │   ├── scheduler.py  # base
│   │   └── # each implementation
│   ├── optimizers
│   │   ├── optimizer.py  # base
│   │   └── # each implementation
│   ├── backends
│   │   ├── __init__.py   # base
│   │   ├── # each implementation with whole components structure 
│   │   ├── cython
│   │   │   ├── # implementation
│   │   │   └── utils
│   │   │       ├── # pyx & pyi files
│   │   │       ├── base.pyi  # shared py interface
│   │   │       └── base.pyd  # shared pyx interface
│   │   └── pycuda
│   │       ├── # implementation
│   │       └── utils
│   │           ├── # cu files
│   │           ├── memory_allocation.py
│   │           └── tensor_array.py
│   ├── datasets
│   │   ├── __init__.py  # usable datasets
│   │   ├── abstract
│   │   │   ├── base.py     # typing interface
│   │   │   ├── utils.py    # utility methods
│   │   │   ├── state.py    # state management
│   │   │   ├── init.py     # initialization
│   │   │   ├── repr.py     # representation
│   │   │   └── augment.py  # data augmentation
│   │   ├── archive.py
│   │   ├── memory.py
│   │   ├── folder.py
│   │   ├── synthetic.py
│   │   └── # each implementation
```
### Support modules
```
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
│       ├── parser.py
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

# Planned
_Things to do_

- Migrate `libs/{cuda,cudadrv,cudart}` to `cuda-bindings` (and/or `nvidia-cuda-runtime-cu12`)
- Migrate `libs/nccl` to `nvidia-nccl-cu12`
- Migrate `libs/cudnn` to `nvidia-cudnn-cu12`
- Migrate `libs/cublas` to `nvidia-cublas-cu12`
- Add PyCUDA parameter quantization (operate on `model.dtype`, weights on `model.param_dtype`)
- Add cuDNN graph backend
- Fix NLP support
- Add model tensor parallelism (previously implemented on a prototype)

# Vendoring
Acquire their dependencies, build them and install them with:
```sh
export $(make env | xargs)
make deps build install
```

For specific dependencies, prefix the target with their name, for example:
```sh
make blis-install
```

_Note: `make *-deps` uses Debian-based package names_

# Tests
_Do things work?_

## All
```sh
make test
```

_Note: exhaustive tests are skipped_

## Specific
```sh
python -m unittest pydtnn.tests.${TEST_FILE}.${TEST_CLASS}.${TEST_METHOD}
```

_Note: include `-v` for verbose mode_

## Exhaustive
```sh
python -m unittest pydtnn.tests.conv_2d_cython.Conv2DCythonTestCase
mpirun python -m unittest pydtnn.tests.conv_2d_conv_gemm_long.Conv2DConvGemmLongTestCase.test_forward_backward_multiple_params
```

## GPU
```sh
python -m unittest pydtnn.tests.model_gpu
```

# Publishing
_Making things public_

## Cleanup
```sh
make format lint
git commit -am cleanup
git push
```

## Build
```sh
make build test
```

## Publish
```sh
twine upload ./build/pydtnn/pydtnn-*
```