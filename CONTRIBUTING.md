# Guidelines
- **Don't re-export symbols**
  Always import modules from it true location, don't use `__init__.py` or modules to create shorthands.
  It can easly cause circular imports, and problems for type-checker and IDEs.

- **Don't use star imports**
  Always import a concrete element or a module, don't plute the global environment.
  It can cause problems for type-checker and IDEs, especialy if multiple are present.

- **For randomness use `pydtnn.utils.random`**
  The module ensure consistent random partterns across multi-threaded applications.
  Using the global `random` or `numpy.random` modules makes seeding useless on multi-thread contexts.

- **Prefer modules without `__init__.py`**
  Modules construction should be avoided.
  It can easly cause circular imports if combined with submodules.

- **Every CUDA's kernel must have a different name**
  If two kernels have the same name, CUDA will not identify the correct function and will throw and error.

- **Test changes in every backend (cpu, gpu, ...)**
  Be careful: any change made in the abstract class in order to improve one backend may break something in the other backends.

- **Use numpy's functions over its operands**
  Even if technically they are the same, sometimes, `numpy.add(a, b, out=a)` has better precision than `a += b`.

- **Ensure ``parser`` and ``README`` are in-sync**
  When adding, modifing or deleting options, check changes are reflected on both sources.

# Knowledge
- `enable_gpu` changes the backed from CPU to GPU
- `enable_gpudirect` changes where data is stored, from CPU in `ndarray` to GPU in `GPUArray`, and requires `enable_gpu`
- `enable_nccl` changes where reductions are made, from CPU with `MPI` to GPU with `NCCL`, and requries `enable_gpudirect`
- `encryption` requies `NCCL` to be off, it it is on, encryption will be skipped.

# Planned
- Extract GPU `SourceModule` to `.cu` files.
- Replace `print` statments with `logger` calls.
- Explore `TenSEAL` serialization preformance.
- Explore `net-queue`'s `TCP+TLS` preformance.
- Explore `pympi`'s 1, 2 (slowest) & 3 client preformance.