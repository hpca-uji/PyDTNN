# Guidelines
- Don't re-export symbols from `__init__.py` or other, instead modules use from their true location.
- Don't use star imports. It can cause problems for type-checker and IDEs, especialy if multiple are present.
- Use `pydtnn.utils.random` for randomness, other generator are not multi-thread aware.
- If plausible do not define `__init__.py` in modules. It can easly cause circular imports.
- Every CUDA's kernel must have a different name. If two kernels have the same name, CUDA will not identify the correct function.
- Test changes in every backend (cpu, gpu, ...). Changes base clases may have unexpected changes in some backends.
- Use `numpy`'s functions over its operands  versions. Sometimes they over better precision, even if theoretically identical.
- Ensure `README.rst` and `parser.py` are in-sync. When adding, modifing or deleting options, check changes are reflected on both sources.

# Knowledge
- `enable_gpu` changes the backed from CPU to GPU.
- `enable_gpudirect` changes where data is stored, from CPU in `ndarray` to GPU in `GPUArray`, and requires `enable_gpu`.
- `enable_nccl` changes where reductions are made, from CPU with `MPI` to GPU with `NCCL`, and requries `enable_gpudirect`.
- `encryption` requies `NCCL` to be off, it it is on, encryption will be skipped.
- If using `conda` and `pip install --config-settings editable_mode=compat -e .` errors with `no such option: --config-settings`,
  deactivate all environments and then reactivate only the one you want.

# Planned
- Move common code of `.pyi` and `.pyx` to a shared module.
- Extract GPU `SourceModule` to `.cu` files.
- Rework the `Conv2D` variant implementation.
- Rework the layer fusion implementation.
- Replace `print` statments with `logger` calls.
- Explore `TenSEAL`'s serialization preformance.
- Explore `net-queue`'s `TCP+TLS` preformance.
- Explore `pympi`'s 1, 2 (slowest) & 3 client preformance.