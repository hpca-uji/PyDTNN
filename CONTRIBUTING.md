# Guidelines
- Do not re-export symbols (ie from `__init__.py`), always import from the original module.
- Avoid star imports (`from x import *`), they break type checking and IDE support.
- Avoid defining `__init__.py` unless strictly necessary (it can easily introduce circular imports).

  ---
- Prefer `numpy` functions over operators when possible (they may offer better numerical precision).
- Do not create arrays using `np.ndarray()`, use explicit initializers such as `np.zeros()`, `np.ones()`, `np.arange()`, or `np.empty()`.
- Avoid magic numbers in `np.transpose(ary, format)`, use `format_transpose(ary, src, dst)` instead.

  ---
- Cython's `.pyx` can be included anywhere, but must be accompanied by a `.pyi` typing interface.

  ---
- Use `pydtnn.utils.random` for random number generation, other generators are not multi-thread aware.

  ---
- Each CUDA kernel must have a unique name, duplicate names will cause incorrect kernel resolution.

  ---
- Test all changes across backends (CPU, GPU, etc.), changes in base classes may introduce backend-specific issues.
- When comparing outputs between layers or models, always copy outputs before passing them to the next layer, some layers perform in-place operations.

  ---
- Keep `README.md` and `parser.py` in sync, any change in options must be reflected in both.

  ---
- Use `from __future__ import annotations` instead of string type annotations, it is more legible and will be default moving forward.

  ---
- If model or components structure changes, update the `Structure` section of this document accordingly.
- If the Model components or components hierarchy structure changes, update the model's `__init__` diagram.

# Knowledge
- In components, `__init__` is used for model-agnostic configuration, and `_model_init` for model specific configuration, and `_post_init` and resource allocation.

  ---
- `--enable-gpudirect` moves data from CPU (`ndarray`) to GPU (`GPUArray`), requires `enable-cudnn`
- `--enable-nccl` moves reductions from CPU (`MPI`) to GPU (`NCCL`), requires `enable-gpudirect`

  ---
- Requires `NCCL` to be disabled (otherwise it will be skipped), typically requires
  `--use-mpi-buffers=False` (crypto libraries usually do not expose buffer access) and
  `--use-blocking-mpi=True` (MPI like `mpi4py` does not support async object reductions)

  ---
- When using `PreallocMemory`, temporary memory in block layers will overlap its child layers, therefore it may be overwritten.

  ---
- If using `conda` and `pip install` fails with `no such option: --config-settings`, deactivate all environments and reactivate only the target environment.

# Structure
## Repository root
```
├── README.md
├── CONTRIBUTING.md
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
│   │   ├── dataset.py  # base
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
- Migrate to `libs/{cuda,cudadrv,cudart}` to `cuda-bindings` (and/or `nvidia-cuda-runtime-cu12`)
- Migrate to `libs/nccl` to `nvidia-nccl-cu12`
- Migrate to `libs/cudnn` to `nvidia-cudnn-cu12`
- Migrate to `libs/cublas` to `nvidia-cublass-cu12`
- Add PyCUDA parameter quantization (operate on model.dtype, weights on model.param_dtype)
- Update CuDNN to graph implementation
- Fix NLP support
- Add model tensor parallelism (previously implemented on a prototype)

# Publishing
Dependencies: `gcc patchelf` and `build twine auditwheel`  

## Cleanup sources
```sh
./scripts/utils/run_formatter.sh pydtnn
git add .
git commit -m 'format codebase'
git push
```

## Publish sources
```sh
git checkout master
git merge develop
git push
git checkout develop
```

### Build distribution
```sh
rm -rf ./build/
python -m build --outdir ./build/ --sdist
python -m build --outdir ./build/wheel/ --wheel
python -m auditwheel repair --wheel-dir ./build/ ./build/wheel/*.whl
```

### Publish distribution
```sh
python -m twine upload --repository pypi ./build/pydtnn-*
```