import numpy as np
import numpy.typing as npt

from typing import List, Any, Tuple, Iterator
from genetic import COP, Chromosome, TdChromosome


class Crossover:
    """
    TODO: Add more sophisticated Crossover Methods
    """
    @staticmethod
    def single_point_crossover(cop: COP, cp1: Chromosome, cp2: Chromosome, pc: float) -> Tuple[Chromosome, Chromosome]:
        """
        Simple Single-point Crossover Operation
        """
        co1, co2 = Chromosome.copy_of(cp1), Chromosome.copy_of(cp2)
        # co1, co2 = cp1[:], cp2[:]
        if np.random.random() < pc:
            cpt = np.random.randint(len(cp1.data))
            co1.data[cpt:], co2.data[cpt:] = co2.data[cpt:], co1.data[cpt:]

            cop.evaluate_fitness(co1)
            cop.evaluate_fitness(co2)

        return co1, co2

    @staticmethod
    def two_point_crossover(cop: COP, cp1: Chromosome, cp2: Chromosome, pc: float) -> Tuple[Chromosome, Chromosome]:
        """
        Two-point Crossover Operation
        """
        co1, co2 = Chromosome.copy_of(cp1), Chromosome.copy_of(cp2)
        # co1, co2 = cp1[:], cp2[:]
        if np.random.random() < pc:
            cpt1, cpt2 = np.random.randint(len(cp1.data), size = 2)
            co1.data[cpt1:cpt2+1], co2.data[cpt1:cpt2+1] = co2.data[cpt1:cpt2+1], co1.data[cpt1:cpt2+1]

            cop.evaluate_fitness(co1)
            cop.evaluate_fitness(co2)

        return co1, co2

    @staticmethod
    def uniform_crossover(cop: COP, cp1: Chromosome, cp2: Chromosome, pc: float) -> Tuple[Chromosome, Chromosome]:
        """
        Uniform Crossover Operation
        """
        co1, co2 = Chromosome.copy_of(cp1), Chromosome.copy_of(cp2)
        # co1, co2 = cp1[:], cp2[:]
        if np.random.random() < pc:
            xsize = np.random.randint(len(cp1.data))
            cpt = np.random.choice(range(len(cp1.data)), size=xsize)
            for i in range(len(cp1.data)):
                if i in cpt:
                    co1.data[i], co2.data[i] = co2.data[i], co1.data[i]
            cop.evaluate_fitness(co1)
            cop.evaluate_fitness(co2)
        return co1, co2

    @staticmethod
    def td_crossover(cop: COP, cp1: TdChromosome, cp2: TdChromosome, pc: float):
        """
        Two-dimensional Crossover Operation as described in
        https://downloads.hindawi.com/journals/mpe/2015/906305.pdf
        """
        def horizontal():
            co1 = np.zeros((len(cp1.data), len(cp1.data[0])))
            co2 = np.zeros((len(cp2.data), len(cp2.data[0])))

            cp1_data = np.array(cp1.data)
            cp2_data = np.array(cp2.data)

            co1[:rr, :] = cp1_data[:rr, :]
            co2[:rr, :] = cp2_data[:rr, :]

            co1[rr:rr + 1, :rc + 1] = cp1_data[rr:rr + 1, :rc + 1]
            co2[rr:rr + 1, :rc + 1] = cp2_data[rr:rr + 1, :rc + 1]

            co1[rr:rr + 1, rc + 1:] = cp2_data[rr:rr + 1, rc + 1:]
            co2[rr:rr + 1, rc + 1:] = cp1_data[rr:rr + 1, rc + 1:]

            co1[rr + 1:, :] = cp2_data[rr + 1:, :]
            co2[rr + 1:, :] = cp1_data[rr + 1:, :]

            chrom1, chrom2 = TdChromosome(data=co1.tolist()), TdChromosome(data=co2.tolist())
            return chrom1, chrom2
        def vertical():
            co1 = np.zeros((len(cp1.data), len(cp1.data[0])))
            co2 = np.zeros((len(cp2.data), len(cp2.data[0])))

            cp1_data = np.array(cp1.data)
            cp2_data = np.array(cp2.data)

            co1[:, :rc] = cp1_data[:, :rc]
            co2[:, :rc] = cp2_data[:, :rc]

            co1[:rr + 1, rc:rc + 1] = cp1_data[:rr + 1, rc:rc + 1]
            co2[:rr + 1, rc:rc + 1] = cp2_data[:rr + 1, rc:rc + 1]

            co1[rr + 1:, rc:rc + 1] = cp2_data[rr + 1:, rc:rc + 1]
            co2[rr + 1:, rc:rc + 1] = cp1_data[rr + 1:, rc:rc + 1]

            co1[:, rc + 1:] = cp2_data[:, rc + 1:]
            co2[:, rc + 1:] = cp1_data[:, rc + 1:]

            chrom1, chrom2 = TdChromosome(data=co1.tolist()), TdChromosome(data=co2.tolist())
            return chrom1, chrom2

        co1, co2 = TdChromosome.copy_of(cp1), TdChromosome.copy_of(cp2)

        if np.random.random() < pc:
            rr = np.random.randint(len(cp1.data))
            rc = np.random.randint(len(cp1.data[0]))

            if np.random.random() > 0.5:
                co1, co2 = horizontal()
            else:
                co1, co2 = vertical()
            cop.evaluate_fitness(co1)
            cop.evaluate_fitness(co2)
        return co1, co2


class Mutation:
    """
    TODO: Add more sophisticated Mutation methods
    """
    @staticmethod
    def single_swap_mutate(cop: COP, c: Chromosome, pm: float) -> Chromosome:
        """
        Simple Single-swap Mutation Operation
        """
        if np.random.random() < pm:
            p1, p2 = np.random.choice(range(len(c.data)), 2, replace=False)
            c.data[p1], c.data[p2] = c.data[p2], c.data[p1]
            cop.evaluate_fitness(c)
        return c

    @staticmethod
    def shuffle_mutate(cop: COP, c: Chromosome, pm: float) -> Chromosome:
        """
        Shuffle Mutation Operation
        """
        if np.random.random() < pm:
            pmc = 1 / len(c.data)
            for i in range(len(c.data)):
                if np.random.random() < pmc:
                    j = np.random.choice(range(len(c.data)))
                    c.data[i], c.data[j] = c.data[j], c.data[i]
            cop.evaluate_fitness(c)
        return c

    @staticmethod
    def drop_mutate(cop: COP, c: Chromosome, pm: float) -> Chromosome:
        """
        Drop Mutation Operation
        """
        if np.random.random() < pm:
            cand = []
            for i in range(len(c.data)):
                if c.data[i] != -1:
                    cand.append(i)
            drop = np.random.choice(cand)
            c.data[drop] = -1
            cop.evaluate_fitness(c)
        return c

    @staticmethod
    def td_mutate(cop: COP, c: TdChromosome, pm: float):
        """
        Two-dimensional Mutation Operation as described in
        https://downloads.hindawi.com/journals/mpe/2015/906305.pdf
        """
        if np.random.random() < pm:
            rr = np.random.randint(len(c.data))
            rc = np.random.randint(len(c.data[0]))
            rrp, rcp = rr, rc
            while rrp == rr and rcp == rc:
                rrp = np.random.randint(len(c.data))
                rcp = np.random.randint(len(c.data[0]))

            c.data[rr][rc], c.data[rrp][rcp] = c.data[rrp][rcp], c.data[rr][rc]
            cop.evaluate_fitness(c)

        return c


class Selection:
    """
    TODO: Add more sophisticated selection methods (e.g. Tournament, Elitism)
    Need to look more into stochastic ranking: https://www.cs.bham.ac.uk/~xin/papers/published_tec_sep00_constraint.pdf
    """
    @staticmethod
    def rank_selection(population: List[Chromosome]) -> Iterator[Chromosome]:
        """
        Linear Rank-based Selection
        """
        ranked = sorted(population, key=lambda x: x.fitness)
        rank_sum = len(ranked) * (len(ranked) + 1) / 2
        distribution = [i / rank_sum for i in range(1, len(ranked) + 1)]
        for i in range(len(ranked)):
            yield ranked[np.random.choice(len(ranked), p=distribution)]

    @staticmethod
    def fitness_selection(population: List[Chromosome]) -> Iterator[Chromosome]:
        """
        Linear Fitness-based Selection
        """
        ranked = sorted(population, key=lambda x: x.fitness)
        rank_sum = len(ranked) * (len(ranked) + 1) / 2
        distribution = [i / rank_sum for i in range(1, len(ranked) + 1)]
        for i in range(len(ranked)):
            yield ranked[np.random.choice(len(ranked), p=distribution)]
    # @staticmethod
    # def rw_selection(population: List[Any], cop: COP):
    #     """
    #     Roulette-wheel Selection (Weighted Random Selection) based on
    #     https://stackoverflow.com/questions/2140787/select-k-random-elements-from-a-list-whose-elements-have-weights
    #     Honestly don't really understand how this works LOLZ
    #     """
    #     fitness_lst = [cop.evaluate_fitness(p) for p in population]
    #     ttl_fitness = sum(fitness_lst)
    #     i = 0
    #     for p in range(len(population) + 1, 0, -1):
    #         x = ttl_fitness * (1 - np.random.random() ** (1. / p))
    #         ttl_fitness -= x
    #         while x > fitness_lst[i]:
    #             x -= fitness_lst[i]
    #             i += 1
    #         fitness_lst[i] -= x
    #         yield population[i]