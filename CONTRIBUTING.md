# Guidelines
- **Don't re-export symbols**

  Always import modules from it true location, don't use `__init__.py` or modules to create shorthands.

  It can easly cause circular imports, and problems for type-checker and IDEs.

- **Don't use star imports**

  Always import a concrete element or a module, don't plute the global environment.

  It can cause problems for type-checker and IDEs, especialy if multiple are present.

- **For randomness use `pydtnn.utils.random`**

  The module ensure consistent random partterns across multi-threaded applications.

  Using the global `random` or `numpy.random` modules makes seeding ustless on multi-thread contexts.

- **Prefer modules without `__init__.py`**

  Modules construction should be avoided.

  It can easly cause circular imports if combined with submodules.

# Coding recommendations
- **Every CUDA's kernel must have a different name**
  If two kernels have the same name, CUDA will not identify the correct function and will throw and error.

- **Always test the new changes in every backend (cpu, gpu, ...)**
  Be careful: any change made in the abstract class in order to improve one backend may broke something in the other backends.

- **In the CPU' backend, it is better to use the numpy's functions**
  Eg: even if technically are the same, sometimes, numpy.add(a, b, out=a) works better "a += b" (where "a" and/or "b" are numpy arrays)

# Planned
- Extract GPU `SourceModule` to `.cu` files.