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

  Modules constuction should be avoided.

  It can easly cause circular imports if combined with submodules.

# Planned
- Extract GPU `SourceModule` to `.cu` files.