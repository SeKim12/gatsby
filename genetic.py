from abc import ABC, abstractmethod
from typing import Any


class COP(ABC):
    @abstractmethod
    def generate_random_config(self):
        pass

    @abstractmethod
    def evaluate_fitness(self, *args):
        pass


class GeneticAlgorithm:
    def __init__(self, N: int, T: int, cop: COP, selection_fn: Any, crossover_fn: Any, mutate_fn: Any, Pc=0.7, Pm=0.1):
        self.N = N
        self.T = T
        self.prob_crossover = Pc
        self.prob_mutation = Pm

        self.cop = cop
        self.select_from = selection_fn
        self.crossover = crossover_fn
        self.mutate = mutate_fn

        self.population = []

        self.best_chromosome = None
        self.best_fitness = float('-inf')

    def initialize_population(self):
        for _ in range(self.N):
            self.population.append(self.cop.generate_random_config())

    def run(self):
        self.initialize_population()

        for t in range(self.T):
            next_generation = []
            gen = self.select_from(self.population, self.cop)
            while len(next_generation) != len(self.population):
                cp1, cp2 = next(gen), next(gen)

                co1, co2 = self.crossover(cp1, cp2, self.prob_crossover)

                next_generation.append(self.mutate(co1, self.prob_mutation))
                next_generation.append(self.mutate(co2, self.prob_mutation))

            self.population = next_generation


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
