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

import traceback
import types
from collections import defaultdict
from contextlib import suppress
from timeit import default_timer as timer
from typing import Hashable, Callable, Tuple, Union, List, Any, Dict, Optional

import numpy as np
from rich import box
from rich.console import Console
from rich.table import Table
from rich.tree import Tree


class _BestOfExecution:
    """
    BestOf execution object
    """

    _names = defaultdict(lambda: 0)
    _longest_name = 0

    def __init__(self, best_of: Optional['BestOf'], execution_id: Optional[Hashable],
                 parent: Optional['_BestOfExecution']):
        self.best_of = best_of
        self.execution_id = execution_id
        self.parent = parent
        self.children: List[_BestOfExecution] = []
        self.problem_sizes = defaultdict(lambda: 0)
        if self.best_of is None:
            self.name = "Execution root"
        else:
            name = self.best_of.name
            index = _BestOfExecution._names[name]
            index += 1
            _BestOfExecution._names[name] = index
            self.name = f"{name} {index:02d}"
        _BestOfExecution._longest_name = max(_BestOfExecution._longest_name, len(self.name))
        if self.parent is not None:
            self.parent.children.append(self)
        self._blocked_by = defaultdict(lambda: defaultdict(lambda: 50))
        self._current_problem_size = None

    def __repr__(self):
        return self.name

    def block_parent(self):
        if not self.parent._is_root:
            self.parent._blocked_by[self.parent._current_problem_size][self] = 50

    def unblock_parent(self):
        if not self.parent._is_root:
            with suppress(KeyError):
                self.parent._blocked_by[self.parent._current_problem_size].pop(self)

    @property
    def is_blocked(self):
        """
        Determines whether this BestOfExecution is blocked or not. To get rid of stale blockers, each block_by counter
        is decreased by one every time this function is called.

        Returns
        -------
        It this BestOfExecution is still blocked.
        """

        # warning: as the blocked_by dictionary could be modified inside the for loop, the keys() generator is first
        #          converted to a list
        for k in list(self._blocked_by[self._current_problem_size].keys()):
            # Decrement blocking counter
            self._blocked_by[self._current_problem_size][k] -= 1
            # If the blocking counter reaches 0, release the corresponding blocker
            if self._blocked_by[self._current_problem_size][k] <= 0:
                self._blocked_by[self._current_problem_size].pop(k)
        return len(self._blocked_by[self._current_problem_size]) > 0

    def set_problem_size(self, problem_size):
        self._current_problem_size = problem_size
        self.problem_sizes[problem_size] += 1

    def print_as_table(self, time_format="6.4f"):
        self.best_of.print_as_table(execution=self, time_format=time_format)

    @property
    def summary(self):
        count = [0] * len(self.best_of.alternatives)
        # Get best_idx for this execution problem sizes
        best_idx = dict((k, self.best_of.best_idx[k])
                        for k in self.problem_sizes.keys() if k in self.best_of.best_idx)
        for idx in best_idx.values():
            count[idx] += 1
        parts = []
        total = len(best_idx)
        for i, alternative in enumerate(self.best_of.alternatives):
            if total == 0:
                parts.append(f"{alternative[0]}: ---")
            else:
                parts.append(f"{alternative[0]}: {(count[i] * 100) / total:.0f}%")
        return "\\[{}]/{}".format(" ".join(parts), total)

    @property
    def max_speedup(self):
        # Get the obtained speedups for this execution problem sizes
        all_speedups = self.best_of.speedups()
        speedups = dict((k, all_speedups[k]) for k in self.problem_sizes.keys() if k in all_speedups)
        if not len(speedups):
            return None
        total = 0
        speedup = 0
        for problem_size in speedups:  # only those with an actual speedup are considered
            count = self.problem_sizes[problem_size]
            speedup += speedups[problem_size] * count
            total += count
        return speedup / total

    @staticmethod
    def _walk_nodes(node: '_BestOfExecution', tree: Tree):
        for child in node.children:
            txt = f"{child.name:{_BestOfExecution._longest_name}s}   {child.summary}"
            max_speedup = child.max_speedup
            if max_speedup:
                txt += f" max speedup: {max_speedup:.1f}"
            branch = tree.add(txt)
            _BestOfExecution._walk_nodes(child, branch)

    def print_report(self):
        tree = Tree("BestOf execution graph")
        _BestOfExecution._walk_nodes(self, tree)
        c = Console(force_terminal=True, width=120)
        c.print(tree)

    # Protected members

    @property
    def _is_root(self):
        return self.parent is None


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
    _current_parents: List[_BestOfExecution] = [_BestOfExecution(best_of=None, execution_id=None, parent=None)]
    _root: _BestOfExecution = _current_parents[0]

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
        self.alternatives = alternatives
        self.get_problem_size = get_problem_size
        self.total_rounds = rounds
        self.stages = stages
        self.prune_after_round = prune_after_round
        self.pruning_speedup = pruning_speedup
        self.best_idx = defaultdict(lambda: -1)
        self.best_name = defaultdict(lambda: 'None')
        self.best_method = defaultdict(lambda: None)
        self.best_pipeline = defaultdict(lambda: None)
        self.total_alternatives = len(self.alternatives)
        # Protected members
        self._previous_round = defaultdict(lambda: 0)
        self._previous_alternative = defaultdict(lambda: 0)
        self._times = defaultdict(self._times_arrays)
        self._stages_times = defaultdict(self._stages_times_arrays)
        self._executions: Dict[List[_BestOfExecution]] = defaultdict(lambda: [])

    def _times_arrays(self) -> List[List]:
        """Returns an array with n empty arrays, where n is the number of alternatives to be evaluated"""
        v = []
        for i in range(self.total_alternatives):
            v.append([])
        return v

    def _stages_times_arrays(self) -> List[List]:
        """
        Returns an array with n arrays with m Nones each, where n is the number of
        alternatives to be evaluated, and m is the number of stages.
        """
        v = []
        for i in range(self.stages):
            v.append([])
        return v

    def _register(self, execution_id) -> _BestOfExecution:
        current_parent = self._current_parents[-1]
        if execution_id in self._executions:
            for execution in self._executions[execution_id]:
                if execution.parent == current_parent:
                    return execution
        current_execution = _BestOfExecution(best_of=self, execution_id=execution_id, parent=current_parent)
        self._executions[execution_id].append(current_execution)
        return current_execution

    @classmethod
    def use_always_the_first_alternative(cls):
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

        # Get _current_execution_id and register this call
        current_execution_id = tuple(traceback.format_list(traceback.extract_stack()))
        current_execution = self._register(current_execution_id)
        # Get stage
        args = list(args)  # Convert args to a list (so that its first element can be removed if in a pipeline)
        stage = int(args.pop(0)) if self.stages > 1 else 0
        assert stage < self.stages, \
            f"The stage number ({stage}) must be less than the specified number of stages ({self.stages})."
        # Should we call the first alternative?
        if self._use_first_alternative:
            first_alternative = self.alternatives[0][1] if self.stages == 1 else self.alternatives[0][1][stage]
            return first_alternative(*args, **kwargs)
        # Get problem size
        problem_size: Hashable = self.get_problem_size(*args, **kwargs)
        current_execution.set_problem_size(problem_size)
        # If best method has been already found, call it and return
        if self.stages == 1:
            best_method: Union[Callable, None] = self.best_method[problem_size]
            if best_method is not None:
                return best_method(*args, **kwargs)
        else:
            best_pipeline: Union[List[Callable], None] = self.best_pipeline[problem_size]
            if best_pipeline is not None:
                return best_pipeline[stage](*args, **kwargs)
        # Block parent until best method is found
        current_execution.block_parent()
        # Set local variables for the given problem size
        previous_alternative = self._previous_alternative[problem_size]
        previous_round = self._previous_round[problem_size]
        if current_execution.is_blocked:
            current_alternative = previous_alternative
            current_round = previous_round
        else:
            evolve = True if self.stages == 1 else False
            round_increment = 1
            if self.stages > 1:
                stages_times = self._stages_times[problem_size]
                # Store the sum of all the stages elapsed times and change to the next alternative only if:
                #  * current stage is 0
                #  * all the stages greater than 0 for the previous alternative have been executed more or
                #    the same number of times of their stage 0, and
                #  * they have been executed at least once.
                stages_that_met_previous_conditions = [1 for x in stages_times[1:]
                                                       if len(x) >= len(stages_times[0]) >= 1]
                if stage == 0 and len(stages_that_met_previous_conditions) == self.stages - 1:
                    medians_per_stage = [np.median(x) for x in stages_times]
                    pipeline_elapsed_time = np.sum(medians_per_stage)
                    # Small caveat: this is performed when stage 0 of the next alternative is executed, it's Ok
                    self._times[problem_size][previous_alternative].append(pipeline_elapsed_time)
                    evolve = True
                    round_increment = len(stages_times[0])
                    # In order to reset self._stages_times[problem_size], problem_size is popped
                    self._stages_times.pop(problem_size)
            if not evolve:
                current_alternative = previous_alternative
                current_round = previous_round
            else:
                # 1) Evolve current alternative and round
                current_alternative = (previous_alternative + 1) % self.total_alternatives
                if current_alternative == 0:
                    current_round = previous_round + round_increment
                else:
                    current_round = previous_round
                # 2) If enough rounds have been performed:
                if current_round >= min(self.prune_after_round, self.total_rounds):
                    best_times = [np.median(x) for x in self._times[problem_size]]
                    min_time = min(best_times)
                    # 2.a) Prune alternatives and ensure that the next alternative is one of remaining ones
                    remaining_alternatives = [i for i, x in enumerate(best_times)
                                              if x <= min_time * self.pruning_speedup]
                    if current_alternative not in remaining_alternatives:
                        for i in remaining_alternatives:
                            if i > current_alternative:
                                current_alternative = i
                                break
                        else:
                            # No remaining alternative is greater than the current one, then a round has been completed
                            current_alternative = remaining_alternatives[0]
                            current_round = previous_round + round_increment
                    # 2.b) Select the best method/pipeline if a new round is going to be performed and either next_round
                    #      is greater than total rounds or there is only one remaining alternative
                    if current_round > previous_round and (current_round >= self.total_rounds
                                                           or len(remaining_alternatives) == 1):
                        current_alternative = best_times.index(min_time)  # first of the minimums
                        self.best_idx[problem_size] = current_alternative
                        if self.stages == 1:
                            (self.best_name[problem_size],
                             self.best_method[problem_size]) = self.alternatives[current_alternative]
                        else:
                            (self.best_name[problem_size],
                             self.best_pipeline[problem_size]) = self.alternatives[current_alternative]
                        # Best method/pipeline set, unblock parent
                        current_execution.unblock_parent()
        # Evaluate current alternative for current round
        BestOf._current_parents.append(current_execution)
        if self.stages == 1:
            alternative = self.alternatives[current_alternative][1]
        else:
            alternative = self.alternatives[current_alternative][1][stage]
        tic = timer()
        output = alternative(*args, **kwargs)
        elapsed_time = timer() - tic
        BestOf._current_parents.pop()
        # Update self._previous_alternative and self._previous_round for the current problem size
        self._previous_alternative[problem_size] = current_alternative
        self._previous_round[problem_size] = current_round
        # If any of the current execution children have not found its best alternative yet, return
        if current_execution.is_blocked:
            if self.stages > 1 and problem_size in self._stages_times:
                # Clear partial times from stages of the same alternative that were not being blocked
                self._stages_times.pop(problem_size)
            return output
        # Record execution time
        if self.stages == 1:
            self._times[problem_size][current_alternative].append(elapsed_time)
        else:
            self._stages_times[problem_size][stage].append(elapsed_time)
        # Return output
        return output

    def best_method_has_been_found(self, *args, **kwargs):
        problem_size: Hashable = self.get_problem_size(*args, **kwargs)
        return problem_size in self.best_idx.keys()

    def medians(self):
        out = {}
        for problem_size, times in self._times.items():
            medians = []
            for i, alternative_times in enumerate(times):
                if len(alternative_times):
                    medians.append(np.median(alternative_times))
                else:
                    medians.append(np.nan)
            out[problem_size] = medians
        return out

    def speedups(self):
        out = {}
        medians = self.medians()
        for problem_size, times in self._times.items():
            best_idx = self.best_idx[problem_size]
            if best_idx != -1:
                out[problem_size] = max(medians[problem_size]) / medians[problem_size][best_idx]
        return out

    def print_as_table(self, execution=None, time_format="6.4f"):
        c = Console(force_terminal=True, width=100)
        caption = self.name if execution is None else execution.name
        t = Table(box=box.HORIZONTALS, show_header=True, header_style="blue", caption=caption)
        t.add_column("size")
        if execution is not None:
            t.add_column("count", justify="right")
        for h in [x[0] for x in self.alternatives]:
            t.add_column(str(h), justify="right")
        t.add_column("speedup", justify="right")
        medians = self.medians()
        speedups = self.speedups()
        for problem_size in self._times.keys():
            if execution is not None and problem_size not in execution.problem_sizes:
                continue
            row_contents = [""] * self.total_alternatives
            for i in range(len(self.alternatives)):
                row_contents[i] = "{0:{1}}".format(medians[problem_size][i], time_format)
            best_idx = self.best_idx[problem_size]
            if best_idx != -1:
                row_contents[best_idx] = "*[bold green]{}[/bold green]".format(row_contents[best_idx])
                row_contents.append("{:.1f}".format(speedups[problem_size]))
            else:
                row_contents.append("")
            if execution is not None:
                row_contents.insert(0, str(execution.problem_sizes[problem_size]))
            row_contents.insert(0, str(problem_size))
            t.add_row(*row_contents)
        c.print(t)

    @staticmethod
    def _walk_nodes_and_print_as_table(node: _BestOfExecution):
        for child in node.children:
            child.print_as_table()
            print()
            BestOf._walk_nodes_and_print_as_table(child)

    @staticmethod
    def print_tables():
        BestOf._walk_nodes_and_print_as_table(BestOf._root)

    @staticmethod
    def print_report():
        BestOf._root.print_report()
        print()
        BestOf.print_tables()
