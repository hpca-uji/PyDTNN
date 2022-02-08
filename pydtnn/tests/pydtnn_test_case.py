import unittest
from abc import ABC

from rich.console import Console

from .common import verbose_test

pydtnn_testcase_status_line = ""
pydtnn_testcase_errors = 0
pydtnn_testcase_failures = 0


class PyDTNNTestCase(unittest.TestCase, ABC):

    def run(self, result=None):
        global pydtnn_testcase_status_line
        global pydtnn_testcase_errors
        global pydtnn_testcase_failures

        console = Console(force_terminal=not verbose_test())
        with console.status(pydtnn_testcase_status_line, spinner="bouncingBar"):
            result = super().run(result)
        errors, failures = len(result.errors), len(result.failures)
        if errors > pydtnn_testcase_errors:
            pydtnn_testcase_status_line += "E"
        elif failures > pydtnn_testcase_failures:
            pydtnn_testcase_status_line += "F"
        else:
            pydtnn_testcase_status_line += "."
        pydtnn_testcase_errors, pydtnn_testcase_failures = errors, failures
        return result

