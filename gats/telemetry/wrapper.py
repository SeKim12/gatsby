from typing import Dict, Any, List
from gats.algorithm import gatsby, genops


class GATSParams:
    def __init__(self):
        self.N = 120
        self.T = 700

        # genops parameters
        self.select_from = genops.selection_rank
        self.crossover = genops.crossover_uniform
        self.mutate = genops.mutate_shuffle
        self.pc = 0.8
        self.pm = 0.08

        # tabu parameters
        self.using_tabu = True
        self.which_tabu = 'flip'

        self.tabu_len = 25
        self.tabu_max_iter = 30
        self.tabu_max_explore = 100

        # convergence parameters
        self.max_fitness = 0
        self.max_repeat = 50
        self.max_time_s = 300
        self.verbose = 1

    def set_params(self, param_dict: Dict):
        for param in param_dict:
            setattr(self, param, param_dict[param])


class GATSTelemetry:
    def __init__(self, cop: gatsby.COP):
        self.cop = cop

    def tune_params(self, params: GATSParams, target: Any, param_range: Any, kw_arr: List[str], iterations=1):
        per_param = {}
        for i in range(len(param_range)):
            p = param_range[i]
            params.set_params({target: p})
            print(f'\nParameter {target} set to {p}\n')
            avg_len_converge = 0
            avg_fitness = 0
            avg_time_converge = 0
            for _ in range(iterations):
                ga = gatsby.Gatsby(
                    N=params.N,
                    T=params.T,
                    cop=self.cop,
                    selection_fn=params.select_from,
                    crossover_fn=params.crossover,
                    mutate_fn=params.mutate,
                    Pc=params.pc,
                    Pm=params.pm,
                    tabu=params.using_tabu,
                    which_tabu=params.which_tabu,
                    max_fitness=params.max_fitness,
                    tabu_list_len=params.tabu_len,
                    tabu_max_iter=params.tabu_max_iter,
                    tabu_max_explore=params.tabu_max_explore,
                    max_repeat=params.max_repeat,
                    max_time_s=params.max_time_s,
                    verbose=params.verbose,
                )

                ga.run()
                avg_fitness += ga.best_chromosome.fitness
                avg_len_converge += len(ga.fitness_hist)
                avg_time_converge += ga.ttl_runtime
            avg_fitness /= iterations
            avg_len_converge /= iterations
            avg_time_converge /= iterations
            per_param[kw_arr[i]] = {"Average Fitness": avg_fitness, "Average Generations": avg_len_converge, "Average Time": avg_time_converge}
        return per_param


