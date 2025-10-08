import os
import platform

import numpy as np

from pydtnn.utils.best_of import BestOf
from pydtnn.utils import print_with_header


class BestOfProfiler:

    def __init__(self, header, best_method):
        self.header = header
        self.best_method: BestOf = best_method

    def __call__(self, *args, **kwargs):
        #
        # First run
        #
        problem_size = self.best_method.get_problem_size(*args, **kwargs)
        print(f"{problem_size}: First run (checking outputs)", sep="", end="")
        outputs = []
        for i in range(self.best_method.total_alternatives):
            outputs.append(self.best_method(*args, **kwargs))
            print(".", sep="", end="")
            if i > 0:
                if isinstance(outputs[0], np.ndarray):
                    name_0 = self.best_method.alternatives[0][0]
                    name_i = self.best_method.alternatives[i][0]
                    assert np.allclose(outputs[0], outputs[-1]), f"{name_0} and {name_i} outputs differ"
        #
        # Rest runs
        #
        print(" ", sep="", end="")
        print(f"Next runs (getting times)", sep="", end="")
        for i in range(0, (self.best_method.total_rounds - 1) * self.best_method.total_alternatives):
            if self.best_method.best_method_has_been_found(*args, **kwargs):
                break
            self.best_method(*args, **kwargs)
            print(".", sep="", end="")
        print()

    def print_results(self):
        c = Console(force_terminal=True)
        #  From IBM OpenMP documentation: If you do not set OMP_NUM_THREADS, the number of processors available is the
        #  default value to form a new team for the first encountered parallel construct.
        import multiprocessing
        num_threads = os.environ.get("OMP_NUM_THREADS", multiprocessing.cpu_count())
        msg = "{}  {}  OMP_NUM_THREADS: {}".format(self.header, platform.node(), num_threads)
        print_with_header(msg)
        self.best_method.print_as_table()
