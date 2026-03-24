# Guidelines
- Don't re-export symbols from `__init__.py` or other, instead modules use from their true location.
- Don't use star imports. It can cause problems for type-checker and IDEs, especially if multiple are present.
- Use `pydtnn.utils.random` for randomness, other generator are not multi-thread aware.
- If plausible do not define `__init__.py` in modules. It can easily cause circular imports.
- Every CUDA's kernel must have a different name. If two kernels have the same name, CUDA will not identify the correct function.
- Test changes in every backend (cpu, gpu, ...). Changes base classes may have unexpected changes in some backends.
- Use `numpy`'s functions over its operands  versions. Sometimes they over better precision, even if theoretically identical.
- Ensure `README.rst` and `parser.py` are in-sync. When adding, modifying or deleting options, check changes are reflected on both sources.
- If a test to compare some layers' outputs of different models is being implemented,
  it is necessary to ensure the copy of those outputs before executing the following layer,
  due there are some layers that operate with their input in-place.
- Don't use `np.ndarray()` to create numpy's array, use a explicit initializer (like `np.zeros()`, `np.ones()`, `np.arange()` or `np.empty()`).
- Don't use `np.transpose(ary, format)` with a magic number, use `format_tranpose(ary, src, dst)` to provide an explicit format.
- The variants of the 2D convolutional layer must be in `conv_2d_variants`. If that is changed,
  it is necessary to reflect this changes in the Conv2D's variable `backend_module_name` and in every backend' variant folder.

# Knowledge
- On components `__init__` is used for configuration, while `initialize` for resource allocations.
- `enable_cudnn` changes the backed from CPU to GPU.
- `enable_gpudirect` changes where data is stored, from CPU in `ndarray` to GPU in `GPUArray`, and requires `enable_cudnn`.
- `enable_nccl` changes where reductions are made, from CPU with `MPI` to GPU with `NCCL`, and requires `enable_gpudirect`.
- `encryption` requires `NCCL` to be off, it it is on, encryption will be skipped.
- `encryption` normally requires `use-mpi-buffers` to be off, as must crypto does not expose buffer access.
  Also the MPI library does not support async object reduces, such as `mpi4py`, `use-blocking-mpi` must be specified.
- If using `conda` and `pip install --config-settings editable_mode=compat -e .` errors with `no such option: --config-settings`,
  deactivate all environments and then reactivate only the one you want.
- Temporal shared memory on block layers may be overwritten by child layers.

# Planned
- Move from `cupy-cuda` package to `cupy` for AMD ROCm support.
- Move `gpu.utils.memory_allocation` from a global namespace to a model instance.
- Move common code of `.pyi` and `.pyx` to a shared module.
- Extract GPU `SourceModule` to `.cu` files.

# Publish
Dependencies: `gcc patchelf` and `build twine auditwheel`

```sh
python -m build --outdir ./dist/
python -m build --outdir ./build/ --wheel
python -m auditwheel repair --wheel-dir ./dist/ ./build/*.whl
python -m twine upload --repository pypi ./dist/*
```