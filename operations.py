import numpy as np
import numpy.typing as npt

from typing import List, Any
from genetic import COP


class Crossover:
    """
    TODO: Add more sophisticated Crossover Methods
    """
    @staticmethod
    def single_point_crossover(cp1: npt.NDArray, cp2: npt.NDArray, pc: float):
        """
        Simple Single-point Crossover Operation
        """
        co1, co2 = cp1[:], cp2[:]
        if np.random.random() < pc:
            cpt = np.random.randint(len(cp1))
            co1[cpt:], co2[cpt:] = co2[cpt:], co1[cpt:]
        return co1, co2

    @staticmethod
    def td_crossover(cp1: npt.NDArray, cp2: npt.NDArray, pc: float):
        """
        Two-dimensional Crossover Operation as described in
        https://downloads.hindawi.com/journals/mpe/2015/906305.pdf
        """
        def horizontal():
            co1 = np.zeros(cp1.shape)
            co2 = np.zeros(cp2.shape)

            co1[:rr, :] = cp1[:rr, :]
            co2[:rr, :] = cp2[:rr, :]

            co1[rr:rr + 1, :rc + 1] = cp1[rr:rr + 1, :rc + 1]
            co2[rr:rr + 1, :rc + 1] = cp2[rr:rr + 1, :rc + 1]

            co1[rr:rr + 1, rc + 1:] = cp2[rr:rr + 1, rc + 1:]
            co2[rr:rr + 1, rc + 1:] = cp1[rr:rr + 1, rc + 1:]

            co1[rr + 1:, :] = cp2[rr + 1:, :]
            co2[rr + 1:, :] = cp1[rr + 1:, :]

            return co1, co2

        def vertical():
            co1 = np.zeros(cp1.shape)
            co2 = np.zeros(cp2.shape)

            co1[:, :rc] = cp1[:, :rc]
            co2[:, :rc] = cp2[:, :rc]

            co1[:rr + 1, rc:rc + 1] = cp1[:rr + 1, rc:rc + 1]
            co2[:rr + 1, rc:rc + 1] = cp2[:rr + 1, rc:rc + 1]

            co1[rr + 1:, rc:rc + 1] = cp2[rr + 1:, rc:rc + 1]
            co2[rr + 1:, rc:rc + 1] = cp1[rr + 1:, rc:rc + 1]

            co1[:, rc + 1:] = cp2[:, rc + 1:]
            co2[:, rc + 1:] = cp1[:, rc + 1:]

            return co1, co2

        if np.random.random() < pc:
            rr = np.random.randint(cp1.shape[0])
            rc = np.random.randint(cp1.shape[1])

            if np.random.random() > 0.5:
                return horizontal()
            else:
                return vertical()
        return cp1, cp2


class Mutation:
    """
    TODO: Add more sophisticated Mutation methods
    """
    @staticmethod
    def single_swap_mutate(c: npt.NDArray, pm: float):
        """
        Simple Single-swap Mutation Operation
        """
        if np.random.random() < pm:
            p1, p2 = np.random.choice(range(len(c)), 2, replace=False)
            c[p1], c[p2] = c[p2], c[p1]
        return c

    @staticmethod
    def td_mutate(c: npt.NDArray, pm: float):
        """
        Two-dimensional Mutation Operation as described in
        https://downloads.hindawi.com/journals/mpe/2015/906305.pdf
        """
        if np.random.random() < pm:
            rr = np.random.randint(c.shape[0])
            rc = np.random.randint(c.shape[1])
            rrp, rcp = rr, rc
            while rrp == rr and rcp == rc:
                rrp = np.random.randint(c.shape[0])
                rcp = np.random.randint(c.shape[1])

            c[rr, rc], c[rrp, rcp] = c[rrp, rcp], c[rr, rc]
        return c


class Selection:
    """
    TODO: Add more sophisticated selection methods (e.g. Tournament, Elitism)
    Need to look more into stochastic ranking: https://www.cs.bham.ac.uk/~xin/papers/published_tec_sep00_constraint.pdf
    """
    @staticmethod
    def rank_selection(population: List[Any], cop: COP):
        """
        Linear Rank-based Selection
        """
        ranked = sorted(population, key=lambda x: cop.evaluate_fitness(x))
        rank_sum = len(population) * (len(population) + 1) / 2
        distribution = [i / rank_sum for i in range(1, len(population) + 1)]
        for i in range(len(ranked)):
            yield ranked[np.random.choice(len(ranked), p=distribution)]

    @staticmethod
    def rw_selection(population: List[Any], cop: COP):
        """
        Roulette-wheel Selection (Weighted Random Selection) based on
        https://stackoverflow.com/questions/2140787/select-k-random-elements-from-a-list-whose-elements-have-weights
        Honestly don't really understand how this works LOLZ
        """
        fitness_lst = [cop.evaluate_fitness(p) for p in population]
        ttl_fitness = sum(fitness_lst)
        i = 0
        for p in range(len(population) + 1, 0, -1):
            x = ttl_fitness * (1 - np.random.random() ** (1. / p))
            ttl_fitness -= x
            while x > fitness_lst[i]:
                x -= fitness_lst[i]
                i += 1
            fitness_lst[i] -= x
            yield population[i]