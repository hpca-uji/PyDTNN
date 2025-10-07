import unittest


try:
    from concurrencytest import ConcurrentTestSuite, fork_for_tests
except ModuleNotFoundError:
    prog = unittest.TestProgram
else:
    class ConcurrentTestProgram(unittest.TestProgram):
        def createTests(self, from_discovery: bool = False, Loader: unittest.TestLoader | None = None) -> None:
            super().createTests(from_discovery, Loader)
            self.test = ConcurrentTestSuite(self.test, fork_for_tests())

    prog = ConcurrentTestProgram

prog(module=None)
