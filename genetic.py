from abc import ABC, abstractmethod
from typing import Any, Optional, Deque, Tuple, Callable, List, Iterator
from collections import deque

import matplotlib.pyplot as plt
import numpy as np


class Chromosome:
    """Chromosome class used in Genetic Algorithm implementation.

    Attributes:
        data (List[Any]): the actual array-like encoding of the problem
        fitness (float): the fitness value of this chromosome
        is_fittest (bool): checks whether chromosome reached max_fitness
    """

    def __init__(self, data=None, fitness=float('-inf')):
        self.data = data or []
        self.fitness = fitness

        self.is_fittest = False

    @staticmethod
    def copy_of(other):
        """Returns deep copy of Chromosome class instance. Use as a copy constructor (i.e. Chromosome.copy_of(other))
        :param other: Chromosome class instance from which we are making the copy
        :return: deep copy of chromosome
        """
        return Chromosome(other.data[:], other.fitness)


class TdChromosome(Chromosome):
    def __init__(self, data: List[List[Any]] = None, fitness=float('-inf')):
        super().__init__()
        self.data = data
        self.fitness = fitness
        self.is_fittest = False

    @staticmethod
    def copy_of(other: 'TdChromosome'):
        return TdChromosome([r[:] for r in other.data], other.fitness)


class COP(ABC):
    """Constrained Optimization Problem interface"""

    @abstractmethod
    def generate_chromosome(self) -> Chromosome:
        """Generate randomly configured chromosome containing actual data and its fitness"""
        pass

    @abstractmethod
    def evaluate_fitness(self, chrom: Chromosome) -> float:
        """Given a chromosome, evaluates its fitness and update/returns c.fitness"""
        pass

    @abstractmethod
    def pretty_print(self, chrom: Chromosome) -> None:
        """Given a chromosome, destructures the chromosome into human-readable format"""
        pass


class GeneticAlgorithm:
    """Runs the genetic algorithm on the given constrained-optimization problem."""
    def __init__(self, N: int, T: int, cop: COP,
                 selection_fn: Callable[[List[Chromosome]], Iterator[Chromosome]],
                 crossover_fn: Callable[[COP, Chromosome, Chromosome, float], Tuple[Chromosome, Chromosome]],
                 mutate_fn: Callable[[COP, Chromosome, float], Chromosome],
                 Pc=0.7, Pm=0.1, max_fitness=float('inf'), tabu=False,
                 tabu_list_len=25, tabu_max_iter=30, tabu_max_explore=200, max_repeat=4, verbose=0):
        """
        :param N: size of population
        :param T: max iterations (generations)
        :param cop: constrained optimization problem (must satisfy COP interface above)
        :param selection_fn: a selection function (must be a Python generator)
        :param crossover_fn: a crossover function
        :param mutate_fn: a mutation function
        :param Pc: probability of crossover
        :param Pm: probability of mutation
        :param max_fitness: maximum possible fitness value for early termination
        :param tabu: whether to turn on TS
        :param tabu_list_len: length of tabu list (recent n moves)
        :param tabu_max_iter: maximum number of moves
        :param tabu_max_explore: maximum number of non-improving neighbors to explore
        :param verbose: verbose mode prints actual chromosome
        """
        self.N = N
        self.T = T
        self.prob_crossover = Pc
        self.prob_mutation = Pm

        self.cop = cop
        self.select_from = selection_fn
        self.crossover = crossover_fn
        self.mutate = mutate_fn

        self.population: List[Chromosome] = []

        self.verbose = verbose

        self.max_fitness = max_fitness

        # Tabu Search Parameters
        self.tabu = tabu
        self.tabu_list_len = tabu_list_len
        self.tabu_max_iter = tabu_max_iter
        self.tabu_max_explore = tabu_max_explore

        self.max_repeat = max_repeat

        # Telemetry
        self.best_chromosome: Optional[Chromosome] = None  # best chromosome across ALL generations
        self.fitness_hist = []  # best fitness per generation

    def initialize_population(self):
        """Initialize N chromosomes in structure defined by COP"""
        for _ in range(self.N):
            self.population.append(self.cop.generate_chromosome())

    def run(self):
        """Run genetic algorithm for T iterations."""
        self.initialize_population()

        conv, count = self.max_fitness, 0

        for t in range(self.T):
            if self.verbose > 0:
                print(f'Starting Generation {t}')
            if self.tabu:
                post_tabu = []
                for p in self.population:
                    improved = self.recursive_tabu_search(p, tl=deque(maxlen=self.tabu_list_len),
                                                          max_iter=self.tabu_max_iter,
                                                          max_explore=self.tabu_max_explore)
                    post_tabu.append(improved)
                    if improved.is_fittest:
                        self.best_chromosome = improved
                        self.fitness_hist.append(0)
                        break

                self.population = post_tabu

            # if tabu search yielded most fit chromosome, no need to check again
            if not self.population[-1].is_fittest:
                self.get_fittest_chromosome()

            if self.best_chromosome.fitness >= self.max_fitness:
                break

            # break early if convergence
            if self.best_chromosome.fitness == conv:
                count += 1
                if count == self.max_repeat:
                    if self.verbose == 1:
                        print(f'value converged for {count} generations, breaking...')
                    break
            else:
                conv, count = self.best_chromosome.fitness, 0

            next_generation = []
            gen = self.select_from(self.population)
            while len(next_generation) != len(self.population):
                cp1, cp2 = next(gen), next(gen)
                co1, co2 = self.crossover(self.cop, cp1, cp2, self.prob_crossover)
                next_generation.append(self.mutate(self.cop, co1, self.prob_mutation))
                next_generation.append(self.mutate(self.cop, co2, self.prob_mutation))

            self.population = next_generation

        self.cop.pretty_print(self.best_chromosome)

    def get_fittest_chromosome(self):
        """Calculates the best chromosome and update self.best_chromosome"""
        best_fitness = float('-inf')
        best_chromosome = None
        for pop in self.population:
            if pop.fitness > best_fitness:
                best_fitness = pop.fitness
                best_chromosome = pop

        self.fitness_hist.append(best_fitness)

        if not self.best_chromosome or best_chromosome.fitness > self.best_chromosome.fitness:
            self.best_chromosome = best_chromosome

        if self.verbose > 0:
            print(f'    Best Fitness: {best_fitness}')
            if self.verbose > 1:
                print(f'    Best Chromsome: {best_chromosome}')

    def plot_fitness(self):
        """Plot fitness history per generation"""
        plt.title("Baseline Genetic Algorithm")
        plt.ylabel("Fitness")
        plt.xlabel("Generation")
        plt.plot(range(len(self.fitness_hist)), self.fitness_hist)
        plt.show()

    def recursive_tabu_search(self, cur: Chromosome, tl: Deque[Any], max_iter: int, max_explore: int):
        """Recursively search neighborhood for max_iter steps.
        Return first-improving neighbor, or continue search on best non-improving neighbor after max_explore

        Args:
            cur: current Chromosome whose neighbor is being explored
            tl: tabu list (recency list) using Python deque. After max_len, the oldest item is popped
            max_iter: number of iterations to run tabu search
            max_explore: number of non-improving neighbors of cur to explore

        Return:
            An improved Chromosome (Lamarckian)
        """
        def random_combination():
            """Random (i, j) combination of i, j in len(cur.data)"""
            while True:
                yield np.random.choice(len(cur.data), 2, replace=False)

        if max_iter == 0:
            return cur
        else:
            count = 0
            best_neighbor = None
            best_fitness = float('-inf')
            move = None

            for i, j in random_combination():
                if tuple(sorted([i, j])) in tl:
                    continue
                neighbor = Chromosome.copy_of(cur)
                neighbor.data[i], neighbor.data[j] = neighbor.data[j], neighbor.data[i]
                self.cop.evaluate_fitness(neighbor)
                if neighbor.fitness > cur.fitness:
                    if neighbor.fitness == self.max_fitness:
                        neighbor.is_fittest = True
                    return neighbor
                else:
                    # otherwise, keep track of best non-improving neighbor
                    if count == max_explore:
                        break
                    count += 1
                    if neighbor.fitness > best_fitness:
                        move = tuple(sorted([i, j]))
                        best_fitness = neighbor.fitness
                        best_neighbor = neighbor
            tl.append(move)
            # search neighborhood of best non-improving neighbor
            return self.recursive_tabu_search(best_neighbor, tl, max_iter - 1, max_explore)

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
