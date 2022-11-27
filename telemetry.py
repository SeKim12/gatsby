from typing import Dict, Any, List

from scheduler import Scheduler
from genetic import GeneticAlgorithm
from operations import Selection, Crossover, Mutation, COP
from courseUtil import CourseBulletin


class GATSParams:
    def __init__(self):
        self.N = 120
        self.T = 700
        self.prob_crossover = 0.8
        self.prob_mutation = 0.08

        self.select_from = Selection.rank_selection
        self.crossover = Crossover.uniform_crossover
        self.mutate = Mutation.shuffle_mutate
        self.max_fitness = 0
        # Tabu Search Parameters
        self.tabu = True
        self.tabu_list_len = 25
        self.tabu_max_iter = 30
        self.tabu_max_explore = 200

    def set_params(self, param_dict: Dict):
        for param in param_dict:
            setattr(self, param, param_dict[param])


class GATSTelemetry:
    def __init__(self, cop: COP):
        self.cop = cop

    def tune_params(self, params: GATSParams, target: Any, param_range: Any, kw_arr: List[str], iterations=1):
        """Iteratively set target parameter to value in param_range
        GA will run with each parameter for iterations and average results
        kw_arr is passed to collect data in readable format (must be in same order as param_range)

        TODO: what data are we collecting? currently taking num generations to convergence
        """
        per_param = {}
        for i in range(len(param_range)):
            p = param_range[i]
            params.set_params({target: p})
            avg = 0.
            for _ in range(iterations):
                ga = GeneticAlgorithm(params.N, params.T, self.cop,
                                      params.select_from, params.crossover, params.mutate,
                                      params.prob_crossover, params.prob_mutation, params.max_fitness,
                                      params.tabu, verbose=1)

                ga.run()
                avg += len(ga.fitness_hist)  # ga.best_chromosome.fitness
            avg /= iterations
            per_param[kw_arr[i]] = avg
        return per_param


if __name__ == '__main__':
    bulletin = CourseBulletin('courses.json')
    cop = Scheduler(bulletin)

    params = GATSParams()
    telem = GATSTelemetry(cop)
    # d = telem.tune_params(params, 'crossover', param_range=[Crossover.uniform_crossover, Crossover.single_point_crossover,
    #                                                     Crossover.two_point_crossover])

    # d = telem.tune_params(params, 'tabu', param_range=[True, False], kw_arr=["Tabu ON", "Tabu OFF"], iterations=3)
    d = telem.tune_params(params, 'N', param_range=[50, 80, 120, 200], kw_arr=['0.3', '0.5', '0.7', '0.9'], iterations=1)

    print('tada:', d)



