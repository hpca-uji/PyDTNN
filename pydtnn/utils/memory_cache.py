
class MemoryCache(dict):
    """
    Dictionary derived class that can use the provided factory function to
    obtain a default value for a missing key. It differs from defaultdict in:

    * The provided factory function receives key as a parameter (which allows
      the generated value to depend on the given key).

    * If disable() is called, the instances of this class will clear their
      already stored values and will not store the next ones.

    """
    _preserve_values = True

    def __init__(self, default_factory=None, **kwargs):
        super().__init__(self, **kwargs)
        self.default_factory = default_factory

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self.default_factory(key)
            if self._preserve_values:
                self[key] = ret
            return ret

    @classmethod
    def disable(cls, update: bool = False):
        cls._preserve_values = False
        if not update:
            return
        import gc
        for obj in gc.get_objects():
            if isinstance(obj, cls):
                obj.clear()

    @classmethod
    def enable(cls):
        cls._preserve_values = True
