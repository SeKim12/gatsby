import numpy as np

from genetic import COP, GeneticAlgorithm, Chromosome
from operations import Selection, Mutation, Crossover
from courseUtil import CourseUtil, CourseBulletin


class Scheduler(COP):
    """
    Four-year scheduler as a constrained-optimization problem
    Access external course data using CourseUtil class
    """

    def __init__(self, course_util: CourseUtil):
        self.course_util = course_util
        self.num_courses = self.course_util.get_num_courses()
        self.num_quarters = 12

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

        TODO: Design better constraints and more sophisticated fitness/penalty functions

        :param chrom: individual chromosome for which we are evaluating its fitness
        :return: fitness value (it also updates the fitness value of chrom instance)
        """
        violations = 0
        preferences = 0

        quarters = {}

        for cindex in range(len(chrom.data)):
            cid = self.cindex_to_cid[cindex]
            assigned_quarter = chrom.data[cindex]

            qid = ''

            if assigned_quarter == -1:
                continue

            if assigned_quarter % 3 == 0:
                qid = "Aut"
            elif assigned_quarter % 3 == 1:
                qid = "Win"
            elif assigned_quarter % 3 == 2:
                qid = "Spr"

            # HC: course must be offered in assigned quarter
            if qid not in self.course_util.get_quarters_offered(cid):
                violations += 1000

            # HC: prerequisites must be satisfied
            for prereq in self.course_util.get_prereqs(cid):
                if prereq in self.cid_to_cindex:
                    pq_cindex = self.cid_to_cindex[prereq]
                    if chrom.data[pq_cindex] == -1 or chrom.data[pq_cindex] > assigned_quarter:
                        violations += 53

            quarters[assigned_quarter] = quarters.get(assigned_quarter, 0) + self.course_util.get_max_units(cid)

        # HC: units must be between 12 <= x <= 18 per quarter
        # Edits: removed the minimum unit requirements for now (It doesn't make sense to fill all quarters with CS courses only)
        for quarter in range(self.num_quarters):
            if quarters.get(quarter, 0) > 18:# or quarters.get(quarter, 0) < 12:
                violations += 1000

        # Check for required core courses
        # CS 103, 109, 106A, 106B, 107, 110, 161
        core_cid = ["CS103", "CS109", "CS106A", "CS106B", "CS107", "CS110", "CS161"]
        core_cindex = [self.cid_to_cindex[cid] for cid in core_cid]
        for cindex in core_cindex:
            if chrom.data[cindex] == -1:
                violations += 500

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
    bulletin = CourseBulletin('courses.json')
    cop = Scheduler(bulletin)
    ga = GeneticAlgorithm(120, 700, cop, Selection.rank_selection, Crossover.uniform_crossover,
                          Mutation.shuffle_mutate, Pc=0.8, Pm=0.08, max_fitness=0, tabu=True, verbose=False)
    ga.run()

    ga.plot_fitness()
