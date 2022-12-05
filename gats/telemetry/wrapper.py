"""Gatsby Wrapper for Data Collection Across Different Parameters.

Tune different parameters and collect data. Must specify config file to set default parameters.
Refer to `tune_params()` method for documetnation on how to change parameters.

Example:
    >>> telem = GATSTelemetry(cop)
    >>> telem.load_config("./path/to/config.yaml")
    >>> telem.tune_params()
"""

from typing import Any, List
from itertools import product
import yaml

from gats.algorithm import gatsby, genops


def stringify(p: Any):
    """Change value of parameter to string.

    Args:
        p: Value of parameter, e.g. if Pc = 0.08, p == 0.08

    Returns:
        String value of the parameter value.
    """
    try:
        return p.__name__
    except AttributeError:
        return str(p)


class GATSTelemetry:
    """Telemetry wrapper for Gatsby algorithm.

    Runs the gatsby algorithm for multiple iterations across different parameters.

    Attributes:
        cop: Constrained Optimization Problem
        params: Must be loaded from a config YAML file using `load_config()`

    """

    def __init__(self, cop: gatsby.COP):
        self.cop = cop
        self.params = None

    def load_config(self, path: str):
        """Load default parameters from YAML file located in `path`.

        To modify default genetic operations, you MUST use the explicit `add_` methods.

        Args:
            path: Path to configuration file

        Returns:
            None, loads default parameters.

        """
        if not path:
            raise RuntimeError("Please specify path to configuration file!")

        with open(path, "r") as file:
            self.params = yaml.safe_load(file)["params"]

        self.params["selection_fn"] = genops.selection_rank
        self.params["crossover_fn"] = genops.crossover_uniform
        self.params["mutate_fn"] = genops.mutate_shuffle

    def dump_config(self):
        """Dump current configuration."""
        printable = {k: stringify(v) for k, v in self.params.items()}
        print(printable)

    def add_selection_fn(self, fn: Any):
        """Change selection function from default.

        `add_` methods can be chained together:

        Example:
            >>> telem.load_config()
            >>> telem.add_selection_fn(...).add_crossover_fn(...)

        Args:
            fn: Selection function from genops.

        Returns:
            None, sets selection function.

        """
        self.params["selection_fn"] = fn
        return self

    def add_crossover_fn(self, fn: Any):
        """Change crossover function from default.

        `add_` methods can be chained together:

        Example:
            >>> telem.load_config()
            >>> telem.add_selection_fn(...).add_crossover_fn(...)

        Args:
            fn: Crossover function from genops.

        Returns:
            None, sets crossover function.

        """
        self.params["crossover"] = fn
        return self

    def add_mutate_fn(self, fn: Any):
        """Change crossover function from default.

        `add_` methods can be chained together:

        Example:
            >>> telem.load_config()
            >>> telem.add_selection_fn(...).add_crossover_fn(...)

        Args:
            fn: Mutation function from genops.

        Returns:
            None, sets mutation function.

        """
        self.params["mutate_fn"] = fn
        return self

    def tune_params(
        self,
        target_params: List[str],
        ranges: List[List[Any]],
        kw_lsts=None,
        iterations=1,
    ):
        """Change parameters across different ranges and collect generation, time, and fitness at convergence.

        If there are 2 target_params, each with a range of 3 values, and iterations == 2,
        then gatsby will run for a total of 18 times.

        Example:
            >>> telem.tune_params(
            >>>     target_params=["Pc", "Pm"],
            >>>     ranges=[[0.8, 0.9, 1.0], [0.08, 0.09, 0.1]]),
            >>>     iterations=2,)

        For the above, the algorithm will run with
            1. (Pc = 0.8, Pm = 0.08) * 2
            2. (Pc = 0.8, Pm = 0.09) * 2
            3. (Pc = 0.8, Pm = 0.1) * 2
            4. (Pc = 0.9, Pm = 0.08) * 2
            ...

        Args:
            target_params: List of parameters to change. The string must exactly match an actual parameter.
            ranges: List of ranges to vary each parameter. The order of ranges must match the target_params.
            kw_lsts: List of keywords to associate with each result. Order must match ranges.
                    If empty, will default to the string value of the parameter value.
            iterations: Number of iterations to run gatsby for each parameter.

        """
        if not self.params:
            raise RuntimeError(
                "\nParameters Not Set! Use `load_config()` with config file path"
            )

        print("\nNOTE: If Not Explicitly Set, Default Genops Will Be Used\n")

        kw_lsts = kw_lsts or [[] for _ in range(len(ranges))]

        for i in range(len(target_params)):
            if target_params[i] not in self.params:
                raise ValueError(f"{target_params[i]} is not a valid Gatsby parameter!")
            if not kw_lsts or len(kw_lsts[i]) != len(ranges[i]):
                kw_lsts[i] = [stringify(r) for r in ranges[i]]

        per_param = {}

        for vals in product(*ranges):
            for i in range(len(target_params)):
                self.params[target_params[i]] = vals[i]
                print(f"{target_params[i]} := {stringify(vals[i])}", end=" // ")

            avg_gen_conv, avg_fitness, avg_time_conv = 0, 0, 0
            for j in range(iterations):
                print(f"Running iteration {j}")
                ga = gatsby.Gatsby(cop=self.cop, **self.params)
                ga.run()
                avg_gen_conv += len(ga.fitness_hist)
                avg_fitness += ga.best_chromosome.fitness
                avg_time_conv += ga.ttl_runtime

            avg_gen_conv /= iterations
            avg_time_conv /= iterations
            avg_fitness /= iterations

            per_param[vals] = {
                "generations": avg_gen_conv,
                "time": avg_time_conv,
                "fitness": avg_fitness,
            }

        print(
            f"\n*****************************************************************************"
            f"\nRan Gatsby With The Following Parameters For {iterations} Iterations Each:"
        )
        for i in range(len(target_params)):
            print(f"\n-->{target_params[i]} := {ranges[i]}")
        print(
            f"****************************************************************************"
        )

        for kw in per_param:
            print(
                f'-->{[target_params[i] + " := " + stringify(kw[i]) for i in range(len(kw))]} : '
                f'\n----> Average Generations Until Convergence : {per_param[kw]["generations"]}'
                f'\n----> Average Time Until Convergence : {per_param[kw]["time"]}'
                f'\n----> Average Maximum Fitness Achieved : {per_param[kw]["fitness"]}'
                f"\n"
            )

        return per_param
