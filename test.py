import numpy as np
import numpy.typing as npt

from genetic import COP, GeneticAlgorithm

from operations import Selection, Mutation, Crossover

CIDS = ["CS106A", "CS106B", "CS103", "CS109", "CS107", "CS111", "CS221", "CS231", "CS161", "CS224N"]

COURSES = {
    "CS106A": {
        "quarters": [0],
        "units": 5,
    },
    "CS106B": {
        "quarters": [0, 1, 2],
        "units": 5,
    },
    "CS103": {
        "quarters": [0, 1, 2],
        "units": 5,
    },
    "CS109": {
        "quarters": [2],
        "units": 4,
    },
    "CS107": {
        "quarters": [1, 2],
        "units": 5,
    },
    "CS111": {
        "quarters": [0, 1, 2],
        "units": 3,
    },
    "CS221": {
        "quarters": [1, 2],
        "units": 4
    },
    "CS231": {
        "quarters": [0, 2],
        "units": 5
    },
    "CS161": {
        "quarters": [1],
        "units": 4
    },
    "CS224N": {
        "quarters": [1, 2],
        "units": 5
    }
}


class ModerateCOP(COP):
    """
    Moderate Constraint Optimization Problem
    Hard Constraints:
        1. Course should be offered at that quarter
        2. Each quarter should be within specified range (5 <= x <= 18)
    Soft Constraints:
        1. Take CS106A (0) in Fall (quarter 0) and CS107 in Spring (quarter 2)

    Each chromosome is 1 dimensional
    Come to think of it, the shape of the chromosome should be independent of the COP
    Maybe should modularize more?
    """
    def __init__(self, num_courses: int):
        self.num_courses = num_courses

    def generate_random_config(self):
        res = []
        for _ in range(self.num_courses):
            res.append(np.random.randint(-1, 3))
        return res

    def evaluate_fitness(self, chrom: npt.NDArray):
        aq_units, wt_units, sp_units = 0, 0, 0
        violations = 0
        preferences = 0
        for course in range(len(chrom)):
            quarter = chrom[course]
            if quarter >= 0:
                unit = COURSES[CIDS[course]]["units"]

                # take CS106A during fall and CS111 during spring
                if course == 0 and quarter == 0: preferences += 1
                if course == len(chrom) - 1 and quarter == 2: preferences += 1
                if course == 5 and quarter == -1: preferences += 1

                if quarter == 0: aq_units += unit
                elif quarter == 1: wt_units += unit
                elif quarter == 2: sp_units += unit
                if quarter not in COURSES[CIDS[course]]["quarters"]: violations += 1
        if aq_units >= 18 or aq_units <= 10: violations += 1
        if wt_units >= 18 or wt_units <= 10: violations += 1
        if sp_units >= 18 or sp_units <= 10: violations += 1

        return preferences - violations


class SimpleCOP(COP):
    """
    Define a simple constrained optimization problem to work with the toy dataset above
    This uses a 2d chromosome. On second thought, though, I think we should keep things simple
    and use a 1d chromosome.
    """
    def __init__(self, num_courses: int):
        self.num_courses = num_courses

    def generate_random_config(self):
        arr = np.arange(self.num_courses).reshape((3, 2))
        for row in range(len(arr)):
            np.random.shuffle(arr[row, :])
        np.random.shuffle(arr)
        return arr
        # return np.random.randint(0, self.num_courses, (3, 2))

    def evaluate_fitness(self, chromosome: npt.NDArray):
        count = 0
        seen = set()
        for quarter in range(chromosome.shape[0]):
            quarter = int(quarter)
            for course in chromosome[quarter, :]:
                course = int(course)
                if course == -1: continue
                if course in seen or quarter not in COURSES[CIDS[course]]["quarters"]:
                    return 0
                seen.add(course)
                count += 1
        return count


if __name__ == '__main__':
    cop = ModerateCOP(num_courses=10)

    # Use rank selection because fitness could be negative.
    # There are normalization techniques for RW selection, but too lazy :(
    ga = GeneticAlgorithm(300, 1200, cop, Selection.rank_selection, Crossover.single_point_crossover, Mutation.single_swap_mutate)

    ga.run()

    best_soln = None
    best_fitness = float('-inf')

    for pop in ga.population:
        if cop.evaluate_fitness(pop) > best_fitness:
            best_soln = pop
            best_fitness = cop.evaluate_fitness(pop)
    # print(ga.population)
    res = []
    print(f'Best Solution w/ fitness {best_fitness}')
    d = {}
    for cid in range(len(best_soln)):
        d[best_soln[cid]] = d.get(best_soln[cid], []) + [CIDS[cid]]

    print(f'Autumn Quarter: {d[0]}')
    print(f'Winter Quarter: {d[1]}')
    print(f'Spring Quarter: {d[2]}')
    print(f'Not Taken: {d[-1]}')
