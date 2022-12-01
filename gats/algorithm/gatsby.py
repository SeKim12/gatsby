"""Gatsby core algorithm and interfaces.

This module contains the core interfaces that must be satisfied to run custom COP on Gatsby,
as well as the core Gatsby algorithm.

Example:
    After defining COP to implement COP interface, running gatsby is as simple as:

        >>> ga = Gatsby(COP, ...)
        >>> ga.run()

    For implementation of COP, check `model/scheduer.py`
"""

import math
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Callable, List, Iterator, Union, Literal
from collections import deque
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


class Chromosome(ABC):
    """Chromosome interface to be used in the Gatsby Algorithm."""
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def fitness(self) -> Union[int, float]:
        """The fitness value of this Chromosome."""
        pass

    @fitness.setter
    @abstractmethod
    def fitness(self, val):
        pass

    @property
    @abstractmethod
    def data(self) -> Union[List[Any], npt.NDArray]:
        """Actual encoded data of this Chromosome. It can be a Python List or a NDArray."""
        pass

    @data.setter
    @abstractmethod
    def data(self, val):
        pass

    @abstractmethod
    def destructure(self) -> Dict[Any, Any]:
        """Destructure Chromosome data to be used for printing, etc."""
        pass

    @abstractmethod
    def deepcopy(self) -> 'Chromosome':
        """Deep copy of all components of this Chromosome."""
        pass

    @abstractmethod
    def interweave1d(self, start_1, end_1, other: 'Chromosome', start_2, end_2) -> Tuple['Chromosome', 'Chromosome']:
        """Equivalent to self.data[start_1: end_1] <-> other.data[start_2: end_2].
        This is to make Chromosome slicing independent of whether data is a Python List or an NDArray.
        Note that this modifies the data of `other` as well."""
        pass

    @abstractmethod
    def swap(self, i, j) -> 'Chromosome':
        """Swap Chromosome data of i-th and j-th index"""
        pass


class COP(ABC):
    """Constrained Optimization Problem interface to be used in the Gatsby Algorithm"""
    @property
    @abstractmethod
    def domain(self) -> List[Any]:
        """List of values that an individual gene can take."""
        pass

    @abstractmethod
    def generate_chromosome(self) -> Chromosome:
        """Generate randomly configured Chromosomes containing actual data and its fitness.
        The actual structure of the Chromosome is thus defined by COP implementations."""
        pass

    @abstractmethod
    def evaluate_fitness(self, chrom: Chromosome) -> Union[int, float]:
        """Evaluate fitness of Chromosome, and update and return fitness attribute."""
        pass

    @abstractmethod
    def pretty_print(self, chrom: Chromosome) -> None:
        """Destructure and Print Chromosome data into Human-readable format."""
        pass


class Gatsby:
    """Genetic Algorithms augmented with Tabu Search."""
    def __init__(self,
                 N: int,
                 T: int,
                 cop: COP,
                 selection_fn: Callable[[List[Chromosome]], Iterator[Chromosome]],
                 crossover_fn: Callable[[COP, Chromosome, Chromosome, float], Tuple[Chromosome, Chromosome]],
                 mutate_fn: Callable[[COP, Chromosome, float], Chromosome],
                 Pc=0.7,
                 Pm=0.1,
                 tabu=False,
                 which_tabu: Literal['swap', 'flip'] = 'flip',
                 tabu_list_len=25,
                 tabu_max_iter=30,
                 tabu_max_explore=100,
                 max_fitness=float('inf'),
                 max_repeat=4,
                 max_time_s=300,
                 verbose=1):
        """
        Attributes:
            N: Size of population.
            T: Maximum number of generations.
            cop: Constrained Optimization Problem.
            selection_fn: Selection genetic operator function that returns a generator type.
            crossover_fn: Crossover genetic operator function that returns two modified Chromosomes.
            mutate_fn: Mutation genetic operator function that returns the modified Chromosome.
            Pc: Probability of crossover.
            Pm: Probability of mutation.
            tabu: Boolean indicator of whether Tabu Search is enabled.
            which_tabu: Type of tabu search to be used. Currently only `swap` and `flip` are implemented.
            tabu_list_len: Length of Tabu List, which is a Recency List.
            tabu_max_iter: Number of iterations to run Tabu Search.
            tabu_max_explore: Number of non-improving neighbors to explore in Tabu Search.
                i.e. O(tabu_max_iter * tabu_max_explore) searches will be performed for each Chromosome.
            max_fitness: Maximum fitness a Chromosome could achieve, upon which the algorithm terminates.
            max_repeat: Maximum number of similar (~=0.2%) fitness values allowed until early termination.
            max_time_s: Maximum time (in seconds) that Gatsby can run for until early termination.
            verbose: For more detailed progress reports.
        """
        self._N = N
        self._T = T
        self._cop = cop

        # genops parameters
        self._select_from = selection_fn
        self._crossover = crossover_fn
        self._mutate = mutate_fn
        self._pc = Pc
        self._pm = Pm

        # tabu parameters
        self._using_tabu = tabu
        self._tabu_fn: Callable[[Chromosome], Chromosome] = self._flip_tabu if which_tabu == 'flip' else self._swap_tabu
        self._tabu_len = tabu_list_len
        self._tabu_max_iter = tabu_max_iter
        self._tabu_max_explore = tabu_max_explore

        # convergence parameters
        self._max_fitness = max_fitness
        self._max_repeat = max_repeat
        self._max_time_s = max_time_s
        self._verbose = verbose

        # telemetry parameters
        self.best_chromosome: Optional[Chromosome] = None  # best chromosome across ALL generations
        self.fitness_hist = []  # best fitness per generation
        self.ttl_runtime = 0  # time spent across ALL generations
        self.time_hist = []  # time spent per generation

        self._population: List[Chromosome] = []

    def _initialize_population(self):
        """Initialize N Chromosomes in the structure defined by COP."""
        for _ in range(self._N):
            self._population.append(self._cop.generate_chromosome())

    def _print_info(self, msg: str, threshold: int):
        """Print progress according to threshold."""
        if self._verbose > threshold:
            print(msg)

    def run(self):
        """Run Gatsby for T iterations.

        Initialize a population of randomly generated Chromosomes.
        For each generation, perform Crossover w. probability Pc to generate new Chromosomes.
        Mutate each Chromosome w. probability Pm and pass them on to next generation.
        If Tabu Search is enabled, improves each Chromosome using selected Tabu Function.

        Gatsby class instance keeps track of per-generation as well as overall history.

        Algorithm terminates when:
            - Maximum generation is reached.
            - Maximum fitness is reached.
            - Converged to a similar fitness value.
            - Algorithm has been running for too long.
        """
        self._initialize_population()
        conv, count = self._max_fitness, 0
        algo_start = time.perf_counter()
        for t in range(self._T):
            gen_start = time.perf_counter()
            if gen_start - algo_start >= self._max_time_s:
                self._print_info(f'\n**********\nTime Limit {self._max_time_s} Seconds Exceeded\n**********\n', 0)
                break
            self._print_info(f'Starting Generation {t}', 0)
            tabu_flag = False
            if self._using_tabu:
                post_tabu = []
                for p in self._population:
                    post_tabu.append(self._tabu_fn(p))
                    # check for premature convergence after tabu search
                    if post_tabu[-1].fitness >= self._max_fitness:
                        self.best_chromosome = post_tabu[-1]
                        self.fitness_hist.append(self._max_fitness)
                        self._print_info(
                            f'\n**********\nGOOD JOB! Achieved Max Fitness After Tabu\n**********\n',
                            -1)
                        tabu_flag = True
                        break
                self._population = post_tabu
            if tabu_flag:
                break

            self._evaluate_population()
            if self.best_chromosome.fitness >= self._max_fitness:
                self._print_info(
                    f'\n**********\nAWESOME SAUCE! Achieved Max Fitness\n**********\n',
                    -1)
                break

            if math.isclose(self.best_chromosome.fitness, conv, rel_tol=0.002):
                count += 1
                if count == self._max_repeat:
                    self._print_info(f'\n**********\nConverged to Sub-optimal Fitness {self.best_chromosome.fitness}\n**********\n', 0)
                    break
            else:
                conv, count = self.best_chromosome.fitness, 0

            next_generation = []
            selector = self._select_from(self._population)
            while len(next_generation) != len(self._population):
                cp1, cp2 = next(selector), next(selector)
                co1, co2 = self._crossover(self._cop, cp1, cp2, self._pc)

                # check for premature convergence after crossover
                if co1.fitness >= self._max_fitness or co2.fitness >= self._max_fitness:
                    self.best_chromosome = co1 if co1.fitness >= self._max_fitness else co2
                    self.fitness_hist.append(self._max_fitness)
                    self._print_info(
                        f'\n**********\nLETS GOOO! Achieved Max Fitness After Crossover\n**********\n',
                        -1)
                    break

                co1, co2 = self._mutate(self._cop, co1, self._pm), self._mutate(self._cop, co2, self._pm)

                # check for premature convergence after mutation
                if co1.fitness >= self._max_fitness or co2.fitness >= self._max_fitness:
                    self.best_chromosome = co1 if co1.fitness >= self._max_fitness else co2
                    self.fitness_hist.append(self._max_fitness)
                    self._print_info(
                        f'\n**********\nBOOYAH! Achieved Max Fitness After Mutation\n**********\n',
                        -1)
                    break

                next_generation.append(co1)
                next_generation.append(co2)

            self._population = next_generation
            self.time_hist.append(time.perf_counter() - gen_start)

            self._print_info(f'==> Generation {t} Completed in {self.time_hist[-1]:.2f} Seconds', 1)

        self.ttl_runtime = time.perf_counter() - algo_start

        self._cop.pretty_print(self.best_chromosome)

    def _evaluate_population(self):
        """Evaluate best Chromosome **per-generation**, and update best Chromosome overall if needed.
        The best fitness value is appended to the history. If this function is not called,
        Fitness values must be explicitly added to maintain full record."""
        best_fitness, best_chromosome = float('-inf'), None
        for pop in self._population:
            if pop.fitness > best_fitness:
                best_fitness = pop.fitness
                best_chromosome = pop

        self.fitness_hist.append(best_fitness)

        if not self.best_chromosome or best_fitness > self.best_chromosome.fitness:
            self.best_chromosome = best_chromosome

        self._print_info(f'==> Best Fitness: {best_fitness}', 0)
        self._print_info(f'==> Best Chromosome: {best_chromosome}', 2)

    def plot_fitness(self):
        """Plot best fitness value across generations."""
        plt.title("Baseline Genetic Algorithm")
        plt.ylabel("Fitness")
        plt.xlabel("Generation")
        plt.plot(range(len(self.fitness_hist)), self.fitness_hist)
        plt.show()

    def plot_time(self):
        """Plot time spent across generations."""
        plt.title("Time Per Generation")
        plt.ylabel("Time")
        plt.xlabel("Generation")
        plt.plot(range(len(self.time_hist)), self.time_hist)
        plt.show()

    def _flip_tabu(self, cur: Chromosome) -> Chromosome:
        """
        Args:
            cur: The Chromosome to improve using Tabu Search.

        Returns:
            The improved Chromosome, or the best non-improving Neighbor.

        Flip Tabu Search randomly selects genes within `cur` and switches the value
        to all possible values within the COP domain. If this results in an improved Chromosome,
        that Chromosome is instantly returned. Otherwise, this process is continued to find the
        Best Non-Improving Chromosome in `cur`'s neighborhood. Search continues for that neighbor
        until `max_iter` is reached.

        Note that this is a First-Improving Neighbor (FIN) Approach, i.e. the search terminates
        the moment a Chromosome whose fitness value exceeds that of `cur` is found.

        Note also that our Tabu Search is Lamarckian, i.e. it replaces the `cur` Chromosome in the population,
        Even though the fitness may be worse-off.
        """
        tabu_list = deque(maxlen=self._tabu_len)
        current = cur
        for _ in range(self._tabu_max_iter):
            count, move, best_neighbor, best_fitness = 0, None, None, float('-inf')
            while count < self._tabu_max_explore:
                ind = np.random.randint(len(current.data))
                for gene in self._cop.domain:
                    move = tuple(sorted([ind, gene]))
                    if move in tabu_list or current.data[ind] == gene:
                        continue
                    neighbor = current.deepcopy()
                    neighbor.data[ind] = gene
                    self._cop.evaluate_fitness(neighbor)
                    if neighbor.fitness > cur.fitness:
                        return neighbor
                    else:
                        count += 1
                        if neighbor.fitness > best_fitness:
                            best_fitness = neighbor.fitness
                            best_neighbor = neighbor
            tabu_list.append(move)
            current = best_neighbor
        return current

    def _swap_tabu(self, cur: Chromosome) -> Chromosome:
        """
        Args:
            cur: The Chromosome to improve using Tabu Search.

        Returns:
            The improved Chromosome, or the best non-improving Neighbor.

        Swap Tabu Search randomly selects two indices within `cur` and swaps their values.
        This tends to result in faster iterations in early generations.
        However, this lacks diversity compared to Flip Tabu Search, and can get stuck
        during later generations.
        """
        def random_combination():
            """Random (i, j) combination of i, j in len(cur.data)"""
            while True:
                yield np.random.choice(len(cur.data), 2, replace=False)
        tabu_list = deque(maxlen=self._tabu_len)
        current = cur
        for _ in range(self._tabu_max_iter):
            count, move, best_neighbor, best_fitness = 0, None, None, float('-inf')
            for i, j in random_combination():
                if tuple(sorted([i, j])) in tabu_list:
                    continue
                neighbor = current.deepcopy()
                neighbor.data[i], neighbor.data[j] = neighbor.data[j], neighbor.data[i]
                self._cop.evaluate_fitness(neighbor)
                if neighbor.fitness > cur.fitness:
                    return neighbor
                else:
                    if count == self._tabu_max_explore:
                        break
                    count += 1
                    if neighbor.fitness > best_fitness:
                        move = tuple(sorted([i, j]))
                        best_fitness = neighbor.fitness
                        best_neighbor = neighbor
            tabu_list.append(move)
            current = best_neighbor
        return current
