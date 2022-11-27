import numpy as np
import numpy.typing as npt

from typing import List, Any, Tuple, Iterator
from genetic import COP, Chromosome, TdChromosome

class Constraint:
    @staticmethod
    def offerings_violation(cop: COP, chrom: Chromosome):
        penalty = 0
        for cindex in range(len(chrom.data)):
            cid = cop.cindex_to_cid[cindex]
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
            if qid not in cop.course_util.get_quarters_offered(cid):
                penalty += 1000
        return penalty

    @staticmethod
    def prereq_violation(cop: COP, chrom: Chromosome):
        penalty = 0
        for cindex in range(len(chrom.data)):
            cid = cop.cindex_to_cid[cindex]
            assigned_quarter = chrom.data[cindex]
            if assigned_quarter == -1:
                continue
            for prereq in cop.course_util.get_prereqs(cid):
                if prereq in cop.cid_to_cindex:
                    pq_cindex = cop.cid_to_cindex[prereq]
                    if chrom.data[pq_cindex] == -1 or chrom.data[pq_cindex] > assigned_quarter:
                        penalty += 53
        return penalty

    @staticmethod
    def units_violation(cop: COP, chrom: Chromosome):
        penalty = 0
        quarters = dict()
        for cindex in range(len(chrom.data)):
            cid = cop.cindex_to_cid[cindex]
            assigned_quarter = chrom.data[cindex]
            if assigned_quarter == -1:
                continue
            quarters[assigned_quarter] = quarters.get(assigned_quarter, 0) + cop.course_util.get_max_units(cid)
        for quarter in range(cop.num_quarters):
            if quarters.get(quarter, 0) > 18:# or quarters.get(quarter, 0) < 12:
                penalty += 1000
        return penalty

    @staticmethod
    def core_violation(cop: COP, chrom: Chromosome):
        penalty = 0
        core_cid = ["CS103", "CS109", "CS106A", "CS106B", "CS107", "CS110", "CS161"]
        core_cindex = [cop.cid_to_cindex[cid] for cid in core_cid]
        for cindex in core_cindex:
            if chrom.data[cindex] == -1:
                penalty += 500
        return penalty

    def track_violation(cop: COP, chrom: Chromosome):
        # AI / HCI / Systems / Theory / Unspecialized
        # Penalty: 250
        penalty = 0
        if not cop.track == "AI":
            return penalty

        # Outsource this part to a different method (track_reqs = load_tracks())
        track_reqs = {
            "A" : set(),
            "B1" : set(),
            "B2" : set(),
            "C" : set()
        }
        track_reqs["A"].add("CS221")
        B1 = ["CS224R", "CS228", "CS229", "CS229M", "CS229T", "CS234", "CS238"]
        for course in B1:
            track_reqs["B1"].add(course)
        B2 = ["CS124", "CS224N", "CS224S", "CS224U", "CS224V"]
        for course in B2:
            track_reqs["B2"].add(course)
        C = ["CS157", "CS205L", "CS230", "CS236", "CS257"]
        for course in C:
            track_reqs["C"].add(course)
        # Outsource up to here
        fulfilled = dict()
        for key in track_reqs:
            fulfilled[key] = False
        for cindex in range(len(chrom.data)):
            cid = cop.cindex_to_cid[cindex]
            assigned_quarter = chrom.data[cindex]
            if assigned_quarter != -1:
                for key in track_reqs:
                    if cid in track_reqs[key]:
                        fulfilled[key] = True
        return penalty

class Objective:
    @staticmethod
    def numcourses_preference(cop:COP, chrom: Chromosome):
        reward = 0
        return reward
    
    @staticmethod
    def core_completion_preference(cop: COP, chrom: Chromosome):
        return 0

    @staticmethod
    def course_level_preference(cop: COP, chrom: Chromosome):
        return 0
    