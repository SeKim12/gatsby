import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Deque, Tuple
from itertools import combinations
from collections import deque

class COP(ABC):
    @abstractmethod
    def generate_random_config(self):
        pass

    @abstractmethod
    def evaluate_fitness(self, *args):
        pass

    @abstractmethod
    def pretty_print(self, *args):
        pass


class GeneticAlgorithm:
    """
    Runs the genetic algorithm on the given constrained-optimization problem.
    """

    def __init__(self, N: int, T: int, cop: COP, selection_fn: Any, crossover_fn: Any, mutate_fn: Any, Pc=0.7, Pm=0.1,
                 max_fitness=float('inf'), tabu=False, verbose=False):
        """
        :param N: size of population
        :param T: max iterations (generations)
        :param cop: constrained-optimization problem (must satisfy COP interface above)
        :param selection_fn: a selection function (must be a Python generator)
        :param crossover_fn: a crossover function
        :param mutate_fn: a mutation function
        :param Pc: probability of crossover
        :param Pm: probability of mutation
        :param max_fitness: maximum possible fitness value for early termination
        :param tabu: whether to turn on TS
        :param verbose:
        """
        self.N = N
        self.T = T
        self.prob_crossover = Pc
        self.prob_mutation = Pm

        self.cop = cop
        self.select_from = selection_fn
        self.crossover = crossover_fn
        self.mutate = mutate_fn

        self.population = []

        self.verbose = verbose

        self.max_fitness = max_fitness

        self.tabu = tabu

        # Telemetry
        self.best_fitness = float('-inf')
        self.best_chromosome = None

        self.fitness_hist = []

    def initialize_population(self):
        """
        Initialize N chromosomes in structure defined by COP
        """
        for _ in range(self.N):
            self.population.append(self.cop.generate_random_config())

    def run(self):
        """
        Run genetic algorithm for T iterations.
        TODO: pass in TS parameters to GA class (Tabu list size, max iterations, etc.)
        """
        self.initialize_population()

        for t in range(self.T):
            if self.tabu:
                post_tabu = []
                for p in self.population:
                    post_tabu.append(self.recursive_tabu_search(p, move=None, best_neighbor=None,
                                                                best_fitness=float('-inf'), tl=deque(maxlen=30),
                                                                max_iter=10))
                self.population = post_tabu
            self.evaluate_population(verbose=self.verbose)
            if self.best_fitness >= self.max_fitness:
                break
            print(f'Starting Generation {t}')
            next_generation = []
            gen = self.select_from(self.population, self.cop)
            while len(next_generation) != len(self.population):
                cp1, cp2 = next(gen), next(gen)

                co1, co2 = self.crossover(cp1, cp2, self.prob_crossover)

                next_generation.append(self.mutate(co1, self.prob_mutation))
                next_generation.append(self.mutate(co2, self.prob_mutation))

            self.population = next_generation

        self.evaluate_population(end=True)

    def evaluate_population(self, verbose=False, end=False):
        """
        Evalute fitness of each chromosome in population and output best result.
        If it is used at the end of GA, then pretty prints the chromosome according to COP.
        """
        self.best_fitness = float('-inf')
        self.best_chromosome = None
        for pop in self.population:
            fitness = self.cop.evaluate_fitness(pop)
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_chromosome = pop

        self.fitness_hist.append(self.best_fitness)

        if end:
            self.cop.pretty_print(self.best_fitness, self.best_chromosome)
        else:
            print(f'    Best Fitness: {self.best_fitness}')
            if verbose:
                print(f'    Best Chromsome: {self.best_chromosome}')

    def plot_fitness(self):
        plt.title("Baseline Genetic Algorithm")
        plt.ylabel("Fitness")
        plt.xlabel("Generation")
        plt.plot(range(len(self.fitness_hist)), self.fitness_hist)
        plt.show()

    def recursive_tabu_search(self, cur: List[Any], move: Optional[Tuple[int, int]], best_neighbor: Optional[List[Any]], best_fitness, tl: Deque[Any], max_iter: int):
        """
        Recursively applies Tabu Search to current chromosome
        This is a Lamarckian operation in the sense that it modifies the chromosome and inserts it into the population
        Currently using First Improving Neighborhood method as outlined in:
            https://www.researchgate.net/publication/255599657_Analysis_of_neighborhood_generation_and_move_selection_strategies_on_the_performance_of_Tabu_Search

        Procedure:
            1. Do a pairwise swap of elements in the current chromosome to generate set of neighbors (one by one)
            2. If we have a fitness-improving neighbor, terminate neighborhood search early, and search the neighbor of that neighbor (FIN approach)
            3. If not, look through all neighbors. At the end, move to the best neighbor, even if non-improving
            4. Record the most recent 30 moves to prevent algorithm from cycling (Tabu List)

        :param cur: the current chromosome whose neighbors we are exploring
        :param move: (i, j) tuple pair indicating the next move (the swap indices that generated cur)
        :param best_neighbor: best neighbor in neighborhood
        :param best_fitness: fitness of best neighbor
        :param tl: tabu list, a Queue w. max_len=30. Deque will automatically pop oldest elements when max_len
        :param max_iter: max number of iterations to do neighborhood search
        :return: the improved chromosome
        """
        if max_iter == 0:
            return cur
        else:
            for i, j in combinations(range(len(cur)), 2):
                if tuple(sorted([i, j])) in tl:
                    continue
                neighbor = cur[:]
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                neighbor_fitness = self.cop.evaluate_fitness(neighbor)
                if neighbor_fitness > self.cop.evaluate_fitness(cur):
                    # if improving neighbor, stop searching current neighborhood and move to neighbor's neighborhood
                    tl.append(tuple(sorted([i, j])))
                    return self.recursive_tabu_search(neighbor, None, None, float('-inf'), tl, max_iter - 1)
                else:
                    # otherwise, keep track of best non-improving neighbor
                    if neighbor_fitness > best_fitness:
                        move = tuple(sorted([i, j]))
                        best_fitness = neighbor_fitness
                        best_neighbor = neighbor
            tl.append(move)
            # search neighborhood of best non-improving neighbor
            return self.recursive_tabu_search(best_neighbor, None, None, float('-inf'), tl, max_iter - 1)


# def stochastic_rank(population: List[Any]):
#     ranked = population[:]
#
#     for i in range(len(ranked)):
#         swapped = False
#         for j in range(len(ranked) - 1):
#             c1, c2 = ranked[j], ranked[j + 1]
#             pen1, pen2 = NQueens.evaluate_penalty(c1), NQueens.evaluate_penalty(c2)
#             if (pen1 == 0 and pen2 == 0) or np.random.random() < 0.4:
#                 if NQueens.evaluate_fitness(c1) > NQueens.evaluate_fitness(c2):
#                     ranked[j], ranked[j + 1] = ranked[j + 1], ranked[j]
#                     swapped = True
#             else:
#                 if pen1 < pen2:
#                     ranked[j], ranked[j + 1] = ranked[j + 1], ranked[j]
#                     swapped = True
#         if not swapped:
#             break
#     return ranked
# #
#
# if __name__ == '__main__':
#     population = []
#     best_soln = None
#     best_fitness = float('-inf')
#
#     # 10 queens
#     nq = NQueens(10)
#     # population size 500
#     ga = SimpleGeneticAlgorithm(30, 10)
#
#     for i in range(30):
#         population.append(nq.generate_random_config())
#
#     for i in range(300):
#         for pop in population:
#             if NQueens.evaluate_fitness(pop) > best_fitness:
#                 best_soln = pop
#                 best_fitness = NQueens.evaluate_fitness(pop)
#
#         population = stochastic_rank(population)
#         # population.sort(key=lambda x: NQueens.evaluate_fitness(x))
#
#         if NQueens.evaluate_fitness(population[-1]) > best_fitness:
#             best_soln = population[-1]
#             best_fitness = NQueens.evaluate_fitness(population[-1])
#
#         # print(f'Iteration {i}, Best Solution => ', best_soln)
#         # print(f'Iteration {i}, Best Fitness => ', best_fitness)
#
#         parents = [None, None]
#         children = []
#         for parent in ga.select(population):
#             if not parents[0] or parents[1]:
#                 parents[0] = parent
#                 parents[1] = None
#             elif not parents[1]:
#                 parents[1] = parent
#                 co1, co2 = ga.crossover(parents[0], parents[1])
#                 children.append(ga.mutate(co1))
#                 children.append(ga.mutate(co2))
#
#         population = children
