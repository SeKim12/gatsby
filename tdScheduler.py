import numpy as np

from genetic import COP, GeneticAlgorithm, Chromosome, TdChromosome
from operations import Selection, Crossover, Mutation
from courseUtil import CourseBulletin, CourseUtil


class TdScheduler(COP):
    def __init__(self, course_util: CourseUtil):
        self.course_util = course_util
        self.num_courses = self.course_util.get_num_courses()
        self.num_quarters = 12
        self.max_course = 6

        self.cids = self.course_util.get_courses()
        self.inds = {}

        for i in range(len(self.cids)):
            self.inds[self.cids[i]] = i

    def ind_to_cid(self, i: int):
        return self.cids[i]

    def cid_to_ind(self, cid: str):
        return self.inds[cid]

    def generate_chromosome(self) -> TdChromosome:
        td_chrom = TdChromosome()
        td_chrom.data = np.random.choice(np.arange(self.num_courses),
                                         size=(self.num_quarters, self.max_course), replace=False).tolist()

        for r in range(len(td_chrom.data)):
            for c in range(len(td_chrom.data[r])):
                if np.random.random() < 0.5:
                    td_chrom.data[r][c] = -1

        self.evaluate_fitness(td_chrom)
        return td_chrom

    def evaluate_fitness(self, chrom: TdChromosome) -> float:
        violations = 0
        units = []
        seen = set()
        core_cid = {"CS103", "CS109", "CS106A", "CS106B", "CS107", "CS110", "CS161"}
        for r in range(len(chrom.data)):
            count = 0
            qid = "Aut" if r % 3 == 0 else "Win" if r % 3 == 1 else "Spr"
            for c in range(len(chrom.data[r])):
                ind = int(chrom.data[r][c])
                if ind == -1: continue
                cid = self.ind_to_cid(ind)

                if cid in seen:
                    violations += 1000
                else:
                    seen.add(cid)

                if cid in core_cid:
                    core_cid.remove(cid)

                if qid not in self.course_util.get_quarters_offered(cid):
                    violations += 1000

                for prereq in self.course_util.get_prereqs(cid):
                    if prereq in self.cids:
                        pq_ind = self.cid_to_ind(prereq)
                        for q in range(r + 1, len(chrom.data)):
                            if pq_ind in chrom.data[q]:
                                violations += 53

                count += self.course_util.get_max_units(cid)
            units.append(count)

        for q in range(len(units)):
            if units[q] > 18:
                violations += 1000

        violations += len(core_cid) * 500
        chrom.fitness = -violations
        return chrom.fitness

    def pretty_print(self, chrom: TdChromosome) -> None:
        for quarter in range(len(chrom.data)):
            print(f'Quarter {quarter} => [ ', end=' ')
            for cl in chrom.data[quarter]:
                cl = int(cl)
                if cl != -1:
                    print(f'{self.ind_to_cid(cl)},', end=' ')
            print(f']')


if __name__ == '__main__':
    bulletin = CourseBulletin('courses.json')
    cop = TdScheduler(bulletin)
    ga = GeneticAlgorithm(400, 700, cop, Selection.rank_selection, Crossover.td_crossover,
                          Mutation.td_mutate, Pc=0.8, Pm=0.08, max_fitness=0, tabu=False, max_repeat=100, verbose=1)
    ga.run()

    ga.plot_fitness()
