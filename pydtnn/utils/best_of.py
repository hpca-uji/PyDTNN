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
import types
from collections import defaultdict
from timeit import default_timer as timer
from typing import Hashable, Callable, Tuple, Union, List, Type, Optional

from rich.console import Console
from rich.table import Table
from rich import box

import numpy as np


class BestOf:
    """
    Automatically executes one of a set of alternatives and eventually selects
    the best one, i.e., the fastest one, for each problem size.

    The alternatives are given as an array of pairs, where each pair is
    formed by:
    * the name of the alternative, and
    * the method to be called when this alternative is selected.

    All the alternative methods have to accept the same parameters and in the
    same order. For those methods with different parameters or parameters order,
    this can be enforced using an intermediate method, or a lambda function,
    that accepts the parameters in the expected order and then calls the actual
    method with its expected parameters.

    Instead of evaluating different methods, it is also possible to evaluate
    different pipelines. For doing this, the total number of stages of the
    pipeline must be specified, and the list of alternatives will provide for
    each alternative pipeline:
    * the name of the alternative pipeline, and
    * an array with the method to be called for each stage of this pipeline.

    To be able to compute the problem size of a given call, a method must be
    provided that returns the problem size as a hashable object. This
    method should accept the same parameters and in the same order as the
    methods that are going to be evaluated.
    """

    _use_first_alternative: bool = False
    _current_parents: List['BestOf'] = []
    _root: List['BestOf'] = []

    def __init__(self,
                 name: str,
                 alternatives: List[Tuple[str, Union[Callable, List[Callable]]]],
                 get_problem_size: Callable[..., Hashable],
                 rounds: int = 10,
                 pruning_speedup: float = 10.0,
                 prune_after_round: int = 4,
                 stages: int = 1):
        # Check parameters constraints
        assert stages >= 1, "Stages must be greater or equal to one."
        assert rounds >= 1, "Rounds must be greater or equal to one."
        assert pruning_speedup > 1, "Pruning speedup must be greater than one."
        if stages == 1:
            for a in alternatives:
                assert type(a[1]) in (types.FunctionType, types.LambdaType, types.BuiltinFunctionType), \
                    f"Expected a function for the '{a[0]}' alternative, got a '{type(a[1])}'."
        else:
            for a in alternatives:
                assert type(a[1]) in (list, tuple), \
                    f"Expected a list with the methods to be called for each stage of the '{a[0]}' pipeline."
                assert len(a[1]) == stages, \
                    f"Expected {stages} methods for the '{a[0]}' pipeline, received {len(a[1])}."
                for i, m in enumerate(a[1]):
                    assert type(m) in (types.FunctionType, types.LambdaType), \
                        f"Expected a function for stage {i} of the '{a[0]}' pipeline alternative."
        # Assign its initial value to each property
        self.name = name
        if stages == 1:
            self.pipeline_alternatives: Optional[List[Tuple[str, List[Callable]]]] = None
            self.alternatives: List[Tuple[str, Callable]] = alternatives
        else:
            self.pipeline_alternatives: Optional[List[Tuple[str, List[Callable]]]] = alternatives
            self.alternatives: List[Tuple[str, Callable]] = []
            for a in alternatives:
                for method in a[1]:
                    self.alternatives.append((a[0], method))
        self.get_problem_size = get_problem_size
        self.total_rounds = rounds * stages
        self.stages = stages
        self.prune_after_round = prune_after_round * stages
        self.pruning_speedup = pruning_speedup
        self.best_idx = defaultdict(lambda: -1)
        self.best_name = defaultdict(lambda: 'None')
        self.best_method = defaultdict(lambda: None)
        self.best_pipeline = defaultdict(lambda: None)
        self.current_round = defaultdict(lambda: 0)
        self.current_alternative = defaultdict(lambda: 0)
        self.total_alternatives = len(alternatives)  # = len(self.alternatives) // self.stages
        self.times = defaultdict(self._times_arrays)
        self.children: List[BestOf] = []

    def _times_arrays(self):
        """Returns an array with n empty arrays, where n is the number of alternatives to be evaluated"""
        v = []
        for i in range(self.total_alternatives):
            v.append([])
        return v

    def _register(self):
        # Warning: The same instance can be registered on different parts. It is not possible to use
        #          an instance property to avoid the next query.
        local_root = self._current_parents[-1].children if len(self._current_parents) else self._root
        if self not in local_root:
            local_root.append(self)

    @classmethod
    def always_call_the_first_alternative(cls):
        """
        Forces all BestOf classes to always call the first alternative,
        deactivating any competition among the different alternatives.
        """
        cls._use_first_alternative = True

    def __call__(self, *args, **kwargs):
        """
        Each time this instance is called, it will call one of the different
        methods provided as alternatives. The received parameters will be passed
        to this method and its output will be returned.

        If a pipeline is being evaluated (stages > 1), the first parameter must
        provide the current stage, and the method corresponding to that stage of
        one of the given alternatives will be executed.

        Also, the execution time for a given problem size will be recorded and,
        eventually, the best method for a given problem size will be determined.

        Parameters
        ----------
        args : array
            Array of arguments to be passed to the method currently being
            evaluated. If the number of stages is greater than one, the first
            argument must be the stage that should be executed. In this case,
            this first argument will be removed from the array of arguments
            passed to the evaluated method.

        kwargs : dictionary
            Dictionary of arguments to be passed to the method currently being
            evaluated.

        Returns
        -------
        The output returned by the called method.
        """
        args = list(args)  # Convert args from a tuple to a list
        stage = args.pop(0) if self.stages > 1 else 0
        assert stage < self.stages, \
            f"The stage number ({stage}) must be less than the specified stages ({self.stages})."
        if self._use_first_alternative:
            return self.alternatives[stage][1](*args, **kwargs)
        problem_size: Hashable = self.get_problem_size(*args, **kwargs)
        if self.stages == 1:
            best_method: Union[Callable, None] = self.best_method[problem_size]
            if best_method is not None:
                return best_method(*args, **kwargs)
        else:
            best_pipeline: Union[List[Callable], None] = self.best_pipeline[problem_size]
            if best_pipeline is not None:
                return best_pipeline[stage](*args, **kwargs)
        # Register this call
        self._register()
        # Set local variables for the given problem size
        current_alternative = self.current_alternative[problem_size]
        current_round = self.current_round[problem_size]
        # Evaluate current alternative for current round
        self._current_parents.append(self)
        tic = timer()
        output = self.alternatives[current_alternative * self.stages + stage][1](*args, **kwargs)
        elapsed_time = timer() - tic
        self._current_parents.pop()
        # Stop here if any of our children have not found its best alternative yet
        for child in self.children:
            if not child.best_method_has_been_found(*args, **kwargs):
                return output
        # If all our children have found their best alternative, record this execution and proceed with the evaluation
        if stage == 0:
            self.times[problem_size][current_alternative].append(elapsed_time)
        else:
            self.times[problem_size][current_alternative][-1] += elapsed_time
        current_alternative = (current_alternative + 1) % self.total_alternatives
        if current_alternative == 0:
            current_round += 1
        if current_round >= min(self.prune_after_round, self.total_rounds):
            best_times = [np.median(x) for x in self.times[problem_size]]
            min_time = min(best_times)
            alternatives_below_pruning_speedup = [x for x in best_times if x <= min_time * self.pruning_speedup]
            if current_round == self.total_rounds or len(alternatives_below_pruning_speedup) == 1:
                # Select best alternative
                self.best_idx[problem_size] = best_times.index(min_time)  # first of the minimums
                if self.stages == 1:
                    (self.best_name[problem_size],
                     self.best_method[problem_size]) = self.alternatives[self.best_idx[problem_size]]
                else:
                    (self.best_name[problem_size],
                     self.best_pipeline[problem_size]) = self.pipeline_alternatives[self.best_idx[problem_size]]
            else:
                # Discard those alternatives with a slow down greater than pruning_speedup
                for i in range(current_alternative, len(best_times)):
                    if best_times[i] <= min_time * self.pruning_speedup:
                        current_alternative = i
                        break
        self.current_alternative[problem_size] = current_alternative
        self.current_round[problem_size] = current_round
        return output

    def best_method_has_been_found(self, *args, **kwargs):
        problem_size: Hashable = self.get_problem_size(*args, **kwargs)
        return problem_size in self.best_idx.keys()

    def print_as_table(self, time_format="6.4f"):
        c = Console(force_terminal=True)
        t = Table(box=box.HORIZONTALS, show_header=True, header_style="blue")
        t.add_column("size")
        for h in [x[0] for x in self.alternatives]:
            t.add_column(str(h), justify="right")
        t.add_column("speedup", justify="right")
        for problem_size, times in self.times.items():
            medians = []
            row_contents = [""] * self.total_alternatives
            for i, alternative_times in enumerate(times):
                medians.append(np.median(alternative_times))
                row_contents[i] = "{0:{1}}".format(medians[-1], time_format)
            best_idx = self.best_idx[problem_size]
            if best_idx != -1:
                row_contents[best_idx] = \
                    "*[bold green]{}[/bold green]".format(row_contents[best_idx])
                row_contents.append("{:.1f}".format(max(medians) / medians[best_idx]))
            else:
                row_contents.append("")
            row_contents.insert(0, str(problem_size))
            t.add_row(*row_contents)
        c.print(t)
