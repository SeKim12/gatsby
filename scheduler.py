import numpy as np

from genetic import COP, GeneticAlgorithm, Chromosome
from operations import Selection, Mutation, Crossover
from courseUtil import CourseUtil, CourseBulletin
from constraints import Constraint, Objective
import argparse

class Scheduler(COP):
    """
    Four-year scheduler as a constrained-optimization problem
    Access external course data using CourseUtil class
    """

    def __init__(self, course_util: CourseUtil, track: str):
        self.course_util = course_util
        self.num_courses = self.course_util.get_num_courses()
        self.num_quarters = 12
        self.track = track
        self.cindex_to_cid = self.course_util.get_courses()
        self.cid_to_cindex = {}

        for i in range(len(self.cindex_to_cid)):
            self.cid_to_cindex[self.cindex_to_cid[i]] = i

    def generate_chromosome(self):
        """Each chromosome has the following layout
        [0, -1, 11, 2, -1, ..., -1]
        where each index corresponds to a unique CS course (cindex_to_cid)
        and the value at each index corresponds to a quarter (0 - 11)
        -1 indicates that that course is never taken

        Intuitively, among all the CS courses, it is more likely that our
        optimal solution has a lot more courses that are NOT taken than taken.
        Therefore, we assign -1 to each course w/ p=0.6 as our initial chromosome
        This will make it more likely that we start off at a feasible region
        """
        chrom = Chromosome()
        for _ in range(self.num_courses):
            quarter = np.random.randint(-1, self.num_quarters)
            if np.random.random() < 0.6:
                quarter = -1
            chrom.data.append(quarter)
        self.evaluate_fitness(chrom)
        return chrom

    def evaluate_fitness(self, chrom: Chromosome):
        """Currently, we only have hard constraints. Therefore, maximum fitness is 0.
        HC # 1 - course must be offered in assigned quarter
        HC # 2 - course prerequisites must be satisfied (however, the penalty of violation is lower)
        HC # 3 - units per quarter must be between 12 <= x <= 18
        HC # 4 - core courses must be all taken
        HC # 5 - requirements for the specificed track must be all taken

        SC # 1 - Number of CS courses taken each quarter is at most 3
        SC # 2 - core courses are completed as soon as possible
        SC # 3 - 200/300 level CS courses are not taken too early

        TODO: implement input parser in __init__ 
        TODO: implement soft constraints as objective rather than penalty and apply stochastic ranking
        TODO: change the termination condition in genetic.py

        :param chrom: individual chromosome for which we are evaluating its fitness
        :return: fitness value (it also updates the fitness value of chrom instance)
        """
        violations = 0
        preferences = 0

        violations += Constraint.offerings_violation(self, chrom)
        violations += Constraint.prereq_violation(self, chrom)
        violations += Constraint.units_violation(self, chrom)
        violations += Constraint.core_violation(self, chrom)
        violations += Constraint.track_violation(self, chrom)

        preferences += Objective.numcourses_preference(self, chrom)
        preferences += Objective.core_completion_preference(self, chrom)
        preferences += Objective.course_level_preference(self, chrom)

        chrom.fitness = preferences - violations
        return chrom.fitness

    def pretty_print(self, chrom: Chromosome):
        """
        Given a fitness and the solution, destructures the problem-specific chromosome.
        It is called within the Genetic Algorithm class.

        :param chrom: chromosome to print
        """
        print(f'Best Solution w/ fitness {chrom.fitness}')
        qts = ["AUT1", "WIN1", "SPR1", "AUT2", "WIN2", "SPR2", "AUT3", "WIN3", "SPR3", "AUT4", "WIN4", "SPR4"]
        quarters = {}
        for cindex in range(len(chrom.data)):
            cid = self.cindex_to_cid[cindex]
            assigned_quarter = chrom.data[cindex]

            qid = qts[assigned_quarter] if assigned_quarter >= 0 else "UNASSIGNED"
            quarters[qid] = quarters.get(qid, []) + [cid]

        print(f'AUTUMN 1 => {quarters.get("AUT1", [])}')
        print(f'WINTER 1 => {quarters.get("WIN1", [])}')
        print(f'SPRING 1 => {quarters.get("SPR1", [])}')
        print(f'AUTUMN 2 => {quarters.get("AUT2", [])}')
        print(f'WINTER 2 => {quarters.get("WIN2", [])}')
        print(f'SPRING 2 => {quarters.get("SPR2", [])}')
        print(f'AUTUMN 3 => {quarters.get("AUT3", [])}')
        print(f'WINTER 3 => {quarters.get("WIN3", [])}')
        print(f'SPRING 3 => {quarters.get("SPR3", [])}')
        print(f'AUTUMN 4 => {quarters.get("AUT4", [])}')
        print(f'WINTER 4 => {quarters.get("WIN4", [])}')
        print(f'SPRING 4 => {quarters.get("SPR4", [])}')
        print(f'UNASSIGNED => {quarters.get("UNASSIGNED", [])}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('track', type=str)
    args = parser.parse_args()
    valid_tracks = ["AI", "HCI", "Systems", "Theory", "Unspecialized"]
    if not args.track in valid_tracks:
        print("Invalid Track! Choose from {AI, HCI, Systems, Theory, Unspecialized}.")
    else:
        print("Chosen Track:", args.track)
        bulletin = CourseBulletin('courses.json')
        cop = Scheduler(bulletin, args.track)
        ga = GeneticAlgorithm(120, 700, cop, Selection.rank_selection, Crossover.uniform_crossover,
                          Mutation.shuffle_mutate, Pc=0.8, Pm=0.08, max_fitness=0, tabu=True, verbose=1)
        ga.run()

        ga.plot_fitness()
