#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021 Universitat Jaume I
#
#  PyDTNN is free software: you can redistribute it and/or modify it under the
#  terms of the GNU General Public License as published by the Free Software
#  Foundation, either version 3 of the License, or (at your option) any later
#  version.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT
#  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
#  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
#  License for more details.
#
#  You should have received a copy of the GNU General Public License along
#  with this program. If not, see <https://www.gnu.org/licenses/>.
#

import os
import platform

import numpy as np
from rich.console import Console
from rich.panel import Panel

from pydtnn.utils.best_of import BestOf


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
                if type(outputs[0]) == np.ndarray:
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
        c.print(Panel.fit(msg))
        self.best_method.print_as_table()
