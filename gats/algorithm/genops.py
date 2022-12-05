"""Genetic Operators that can be used in the Gatsby Algorithm.

Crossover, Mutation, Selection operations are defined.
How chromosomes are spliced together are determined by the implementation of
`gatsby.Chromosome` interface.
"""

from typing import List, Iterator
import numpy as np

from gats.algorithm import gatsby


def crossover_single_point(
    cop: gatsby.COP, cp1: gatsby.Chromosome, cp2: gatsby.Chromosome, pc: float
):
    """Perform Single-point Crossover with Probability `pc`.

    Choose random crossover point and interweave.

    Args:
        cop: The COP implementation used to evaluate the fitness of offsprings.
        cp1: Parent Chromosome 1
        cp2: Parent Chromosome 2
        pc: Probability of Crossover

    Returns:
        co1: Offspring Chromosome 1
        co2: Offspring Chromosome 2

    """
    co1, co2 = cp1, cp2
    if np.random.random() < pc:
        co1, co2 = cp1.deepcopy(), cp2.deepcopy()
        cpt = np.random.randint(len(cp1.data))
        co1.interweave1d(cpt, -1, co2, cpt, -1)
        cop.evaluate_fitness(co1)
        cop.evaluate_fitness(co2)
    return co1, co2


def crossover_two_point(
    cop: gatsby.COP, cp1: gatsby.Chromosome, cp2: gatsby.Chromosome, pc: float
):
    """Perform Two-point Crossover with Probability `pc`.

    Choose two random crossover points and interweave [cpt1: cpt2 + 1).

    Args:
        cop: The COP implementation used to evaluate the fitness of offsprings.
        cp1: Parent Chromosome 1
        cp2: Parent Chromosome 2
        pc: Probability of Crossover

    Returns:
        co1: Offspring Chromosome 1
        co2: Offspring Chromosome 2

    """
    co1, co2 = cp1, cp2
    if np.random.random() < pc:
        co1, co2 = cp1.deepcopy(), cp2.deepcopy()
        cpt1, cpt2 = np.random.randint(len(cp1.data), size=2)
        co1.interweave1d(cpt1, cpt2 + 1, co2, cpt1, cpt2 + 1)
        cop.evaluate_fitness(co1)
        cop.evaluate_fitness(co2)
    return co1, co2


def crossover_uniform(
    cop: gatsby.COP, cp1: gatsby.Chromosome, cp2: gatsby.Chromosome, pc: float
):
    """Perform Uniform Crossover with Probability `pc`.

    Choose multiple crossover points at uniform and interweave.

    Args:
        cop: The COP implementation used to evaluate the fitness of offsprings.
        cp1: Parent Chromosome 1
        cp2: Parent Chromosome 2
        pc: Probability of Crossover

    Returns:
        co1: Offspring Chromosome 1
        co2: Offspring Chromosome 2

    """
    co1, co2 = cp1, cp2
    if np.random.random() < pc:
        co1, co2 = cp1.deepcopy(), cp2.deepcopy()
        xsize = np.random.randint(len(cp1.data))
        cpt = np.random.choice(range(len(cp1.data)), size=xsize)
        for i in range(len(cp1.data)):
            if i in cpt:
                co1.interweave1d(i, i + 1, co2, i, i + 1)
        cop.evaluate_fitness(co1)
        cop.evaluate_fitness(co2)
    return co1, co2


def mutate_single_swap(
    cop: gatsby.COP, c: gatsby.Chromosome, pm: float
) -> gatsby.Chromosome:
    """Perform Single-swap Mutation with Probability `pm`.

    Choose two points at random and swap.

    Args:
        cop: The COP implementation used to evaluate the fitness of offsprings.
        c: Chromosome to be mutated
        pm: Probability of Mutation

    Returns:
        Mutated Chromosome.

    """
    if np.random.random() < pm:
        p1, p2 = np.random.choice(range(len(c.data)), 2, replace=False)
        c.swap(p1, p2)
        cop.evaluate_fitness(c)
    return c


def mutate_shuffle(
    cop: gatsby.COP, c: gatsby.Chromosome, pm: float
) -> gatsby.Chromosome:
    """Perform Shuffle Mutation with Probability `pm`.

    Choose random points and shuffle.

    Args:
        cop: The COP implementation used to evaluate the fitness of offsprings.
        c: Chromosome to be mutated
        pm: Probability of Mutation

    Returns:
        Mutated Chromosome.

    """
    if np.random.random() < pm:
        pmc = 1 / len(c.data)
        for i in range(len(c.data)):
            if np.random.random() < pmc:
                j = np.random.choice(range(len(c.data)))
                c.swap(i, j)
        cop.evaluate_fitness(c)
    return c


def mutate_drop(cop: gatsby.COP, c: gatsby.Chromosome, pm: float) -> gatsby.Chromosome:
    """Perform Drop Mutation with Probability `pm`.

    Randomly choose enrolled courses and drop.

    Args:
        cop: The COP implementation used to evaluate the fitness of offsprings.
        c: Chromosome to be mutated
        pm: Probability of Mutation

    Returns:
        Mutated Chromosome.

    """
    if np.random.random() < pm:
        drop = np.random.choice(
            [c.data[i] for i in range(len(c.data)) if c.data[i] != -1]
        )
        c.data[drop] = -1
        cop.evaluate_fitness(c)
    return c


def selection_rank(population: List[gatsby.Chromosome]) -> Iterator[gatsby.Chromosome]:
    """Perform rank selection on population to choose parent Chromosomes.

    Linear Rank Selection works as follows:
        1. Sort the population in ascending fitness value.
        2. Assign probability to each Chromosome in ascending order.
            (e.g. for 4 elements, [0.1, 0.2, 0.3, 0.4])
        3. Choose from population at random according to the distribution set above.

    Args:
        population: The current population of Chromosomes.

    Yields:
        A Chromosome selected at random, biased toward better Chromosomes.

    """
    ranked = sorted(population, key=lambda x: x.fitness)
    rank_sum = len(ranked) * (len(ranked) + 1) / 2
    distribution = [i / rank_sum for i in range(1, len(ranked) + 1)]
    for i in range(len(ranked)):
        yield ranked[np.random.choice(len(ranked), p=distribution)]
