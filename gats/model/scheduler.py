"""Stanford 4 year Course Scheduler as a Constrained Optimization Problem.

This module contains the Course Scheduling problem modeled as a COP.
The `Scheduler` class exposes methods to add both local and global constraints.
Check `add_local_constraint` and `add_global_constraint` for more information.

Generally speaking, COPs wanting to use the Gatsby algorithm should also define
their own chromosome structure that implements the `gatsby.Chromosome` interface.
"""

from typing import Optional, Callable, Any, Dict, List, Union
from collections import defaultdict

import numpy as np
import numpy.typing as npt

from gats.algorithm import gatsby
from gats.data import util


def to_qid(q: int) -> str:
    """Convert quarter index to quarter ID.

    Args:
        q: A quarter index in [-1, 12)

    Returns:
        A quarter ID in ["Aut", "Win", "Spr"]
    """
    if q == -1:
        return "NA"
    return "Aut" if q % 3 == 0 else "Win" if q % 3 == 1 else "Spr"


class SchedChrom1d(gatsby.Chromosome):
    """Chromosome implementation used in the Scheduler COP"""
    def __init__(self):
        """
        The structure of the Chromosome is [-1, 0, 1, 11, ...], where each index corresponds to a unique CS course,
        and each value corresponds to a quarter `in` [-1, 12) that the CS course will be scheduled.
        Note that -1 indicates that the course is not taken.

        Attributes:
            data: A NDArray encoding of Course Schedule data.
            fitness: Fitness value of Chromosome.
        """
        self._data: Optional[npt.NDArray] = None
        self._fitness = float('-inf')

    @property
    def fitness(self):
        return self._fitness

    @fitness.setter
    def fitness(self, val: float):
        self._fitness = val

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, val: npt.NDArray):
        self._data = val

    def destructure(self):
        """Destructure Chromosome into dictionary.

        Examples:
            >>> c.destructure()
            {-1: ["CS193P", ...], 0: ["CS106A", ...]}
        """
        qts = {}
        for idx in range(self._data.shape[0]):
            qts[self._data[idx]] = qts.get(self._data[idx], []) + [idx]
        return qts

    def deepcopy(self):
        clone = SchedChrom1d()
        clone.data = self._data.copy()
        clone.fitness = self._fitness
        return clone

    def interweave1d(self, start_1, end_1, other: 'SchedChrom1d', start_2, end_2):
        self._data[start_1: end_1], other._data[start_2: end_2] = \
            other._data[start_2: end_2], self._data[start_1: end_1]
        return self, other

    def swap(self, i, j):
        self._data[i], self._data[j] = self._data[j], self._data[i]
        return self


class Scheduler(gatsby.COP):
    """Scheduler COP to be solved using Gatsby Algorithm."""
    def __init__(self, course_util: util.CourseParser, track: str):
        """
        Attributes:
            course_util: CourseParser implementation to process Course Information.
            num_courses: Number of courses to schedule across 4 years.
            num_quarters: Number of quarters.
            track: String value of track.
            track_reqs: Dictionary of track requirements by part.
            core_reqs: Set of core requirements.
            domain: [-1, 12)
            cids: List of string course IDs used to map index to cid.
            idx: Dictionary of course IDs to corresponding index.
            loc_cstrt_cnt: Number of local constraints.
            loc_cstrt_idx: Map unique index to local constraint name.
            loc_cstrt_fns: Map unique index to local constraint functions.
            loc_cstrt_pen: Map unique index to local constraint penalties.
            collectors: List of collector functions.
            states: Accumulate local states to be used by global constraints.
            glob_cstrt_cnt: Number of global constraints.
            glob_cstrt_idx: Map unique index to global constraint name.
            glob_cstrt_fns: Map unique index to global constraint function.
            glob_cstrt_pen: Map unique index to global constraint penalties.

        """
        self._course_util = course_util
        self._num_courses = self._course_util.get_num_courses()
        self._num_quarters = 12
        self._track = track
        self._track_reqs = util.load_track(track)
        self._core_reqs = util.load_core()

        self._domain = [i for i in range(-1, self._num_quarters)]

        self._cids = self._course_util.get_courses()
        self._idx = {self._cids[i]: i for i in range(len(self._cids))}

        self._loc_cstrt_cnt = 0
        self._loc_cstrt_idx = {}
        self._loc_cstrt_fns: Dict[int, Callable[[SchedChrom1d, int], Any]] = {}
        self._loc_cstrt_pen = {}

        self._collectors: List[Callable[[SchedChrom1d, int], Any]] = []
        self._states = defaultdict(lambda: defaultdict(int))

        self._glob_cstrt_cnt = 0
        self._glob_cstrt_idx = {}
        self._glob_cstrt_fns: Dict[int, Callable[[SchedChrom1d], Any]] = {}
        self._glob_cstrt_pen = {}

    def _to_idx(self, cid: str) -> int:
        """Map course ID to unique index

        Args:
            cid: Course ID, e.g. "CS106A"

        Returns:
            unique index, e.g. 53
        """
        return self._idx[cid]

    def _to_cid(self, idx: int) -> str:
        """Map course index to course ID

        Args:
            idx: unique index, e.g. 53

        Returns:
            cid, e.g. "CS106A"
        """
        return self._cids[idx]

    @property
    def domain(self):
        return self._domain

    def generate_chromosome(self):
        """Intuitively, a 4-year schedule is not going to include a majority of CS courses.
        Therefore, we initially drop 60% of courses for each Chromosome."""
        cr = SchedChrom1d()
        cr.data = np.random.randint(self._num_quarters, size=self._num_courses)
        cr.data[np.random.rand(*cr.data.shape) < 0.6] = -1
        self.evaluate_fitness(cr)
        return cr

    def stringify_constraints(self):
        """Return a string representation of all constraints defined in the COP"""
        res = []
        for k in self._loc_cstrt_idx:
            res.append(f'LOCAL => {self._loc_cstrt_idx[k]}: Index {k}, Penalty {self._loc_cstrt_pen[k]}')
        for k in self._glob_cstrt_idx:
            res.append(f'GLOBAL => {self._glob_cstrt_idx[k]}: Index {k}, Penalty {self._glob_cstrt_pen[k]}')
        return '\n'.join(res)

    def add_local_constraint(self, name: str, fn: Callable[[SchedChrom1d, int], Any], pen: Union[int, float]):
        """Add local constraint functions and penalties to the COP.

        Args:
            name: Name of the local constraint.
            fn: The constraint function that returns a count of violations.
            pen: The penalty associated per each counted violation.

        Local Constraints are constraints that must look at each particular gene.
        Examples of this would be prerequisite constraints or quarter offering constraints.
        Each local constraint takes in the entire Chromosome, as well as an individual index.
        It is called within a for loop to process violations per each index.

        Refer to `_hc_quarter` to see how local constraint functions are defined,
        `add_default_constraints` to see how constraints are added,
        and `evaluate_fitness` to see how they are used.
        """
        self._loc_cstrt_idx[self._loc_cstrt_cnt] = name
        self._loc_cstrt_fns[self._loc_cstrt_cnt] = fn
        self._loc_cstrt_pen[self._loc_cstrt_cnt] = pen
        self._loc_cstrt_cnt += 1
        return self._loc_cstrt_cnt

    def _add_collector(self, key: str, fn: Callable[[SchedChrom1d, int], Any]):
        """Add collector functions to the COP.

        Args:
             key: The key to the state space allocated for the particular constraint.
             fn: The collector function that updates the state space.

        Collectors are used to collect some information about each index of the Chromosome.
        This information is stored in the `self._state` and is later retrieved by global constraints
        to evaluate global penalties.

        Refer to `_units_collector` to see how collectors are defined,
        `add_global_constraint` to see how they are added,
        and `evaluate_fitness` to see how they are used.
        """
        assert key not in self._states
        self._collectors.append(fn)

    def add_global_constraint(self, name: str, cstrt_fn: Callable[[SchedChrom1d], Any], pen, coll_fn: Optional[Callable[[SchedChrom1d, int], Any]], key=''):
        """Add global constraint functions and penalties to the COP.

        Args:
            name: Name of the global constraint.
            cstrt_fn: The constraint function that returns a count of violations.
            pen: The penalty associated for each counted violation.
            coll_fn: The collector function responsible for collecting local search data.
            key: The key to the state space. If empty, defaults to constraint name.

        Global Constraints are constraints that must look at the entire Chromosome.
        Examples of this would be unit constraints or track requirement constraints.
        Each global constraint takes in the entire Chromosome.
        It retrieves accumulated data collected by its associated collector from the state space.

        Refer to `_glob_hc_unit_count` to see how global constraint functions are defined,
        `add_default_constraints` to see how constraints are added,
        and `evaluate_fitness` to see how they are used.
        """
        self._glob_cstrt_idx[self._glob_cstrt_cnt] = name
        self._glob_cstrt_fns[self._glob_cstrt_cnt] = cstrt_fn
        self._glob_cstrt_pen[self._glob_cstrt_cnt] = pen
        self._glob_cstrt_cnt += 1

        if coll_fn:
            self._add_collector(key or name, coll_fn)

        return self._glob_cstrt_cnt

    def add_default_constraints(self):
        """Add default Hard and Soft Constraints.

        A detailed explanation of each constraint is provided in `evaluate_fitness`.
        """
        self.add_global_constraint("HC: Unit Overload Constraint", self._glob_hc_unit_count, 1000, self._units_collector, "unit_count")
        self.add_global_constraint("HC: Track Requirement Constraint", self._glob_hc_track, 250, self._track_collector, "track")
        self.add_global_constraint("HC: Core Requirement Constraint", self._glob_hc_core, 500, None)
        self.add_global_constraint("SC: Course Overload Constraint", self._glob_sc_course_num, 11, self._course_collector, "course_count")

        self.add_local_constraint("HC: Quarter Offering Constraint", self._hc_quarter, 1000)
        self.add_local_constraint("HC: Unsatisfied Prerequisite Constraint", self._hc_prereq, 53)
        self.add_local_constraint("SC: Taking Hard Courses Too Early Constraint", self._sc_course_sequence, 5)

    def _units_collector(self, chrom: SchedChrom1d, idx: int):
        """Store unit count per each quarter in state space.

        Args:
            chrom: Chromosome being processed.
            idx: Current local index of `chrom` being processed.
        """
        qtr, cid = chrom.data[idx], self._to_cid(idx)
        self._states["unit_count"][qtr] += self._course_util.get_max_units(cid)

    def _glob_hc_unit_count(self, _):
        """Count number of quarters in which unit count is greater than 18."""
        count = 0
        for qtr in self._states["unit_count"]:
            count += int(self._states["unit_count"][qtr] > 18)
        return count

    def _course_collector(self, chrom: SchedChrom1d, idx: int):
        """Store course count per each quarter in state space.

        Args:
            chrom: Chromosome being processed.
            idx: Current local index of `chrom` being processed.
        """
        qtr, cid = chrom.data[idx], self._to_cid(idx)
        self._states["course_count"][qtr] += 1

    def _glob_sc_course_num(self, _):
        """Count number of quarters in which course count is greater than 3."""
        count = 0
        for qtr in self._states["course_count"]:
            count += int(self._states["course_count"][qtr] > 3)
        return count

    def _track_collector(self, _, idx: int):
        """Store whether each part of the track is satisfied state space.

        Args:
            idx: Current local index of `chrom` being processed.
        """
        cid = self._to_cid(idx)
        for k in self._track_reqs:
            if cid in self._track_reqs[k]:
                self._states["track"][k] = 1
            else:
                self._states["track"][k] = self._states["track"][k]

    def _glob_hc_track(self, _):
        """Count number of track parts that were not satisfied"""
        count = 0
        for k in self._states["track"]:
            if self._states["track"][k] != 1:
                count += 1
        return count

    def _glob_hc_core(self, chrom: SchedChrom1d):
        """Count number of core requirements that were not satisfied

        Note that core classes taken later are penalized, but by a smaller amount.
        Normalize each count so that penalty differs.
        """
        count = 0
        for core in self._core_reqs:
            idx = self._to_idx(core)
            if chrom.data[idx] == -1:
                count += 1
            elif chrom.data[idx] > 6:
                count += 0.1
        return count

    def _hc_quarter(self, chrom: SchedChrom1d, idx: int):
        """Check whether course at idx is offered at scheduled quarter.

        Args:
            chrom: Chromosome being processed.
            idx: Current local index of `chrom` being processed.
        """
        c = 0
        qid, cid = to_qid(chrom.data[idx]), self._to_cid(idx)
        if qid != "NA" and qid not in self._course_util.get_quarters_offered(cid):
            c += 1
        return c

    def _hc_prereq(self, chrom: SchedChrom1d, idx: int):
        """Check whether course at idx has satisfied all its prerequisites.

        Args:
            chrom: Chromosome being processed.
            idx: Current local index of `chrom` being processed.
        """
        c = 0
        cid = self._to_cid(idx)
        for pq in self._course_util.get_prereqs(cid):
            if pq in self._idx:
                pq_idx = self._to_idx(pq)
                # if prerequisite course is not taken or is taken at a later quarter
                if chrom.data[pq_idx] == -1 or chrom.data[pq_idx] > chrom.data[idx]:
                    c += 1
        return c

    def _sc_course_sequence(self, chrom: SchedChrom1d, idx: int):
        """Check whether 200/300 courses are taken at earlier quarters.

        Args:
            chrom: Chromosome being processed.
            idx: Current local index of `chrom` being processed.
        """
        qtr, cid = chrom.data[idx], self._to_cid(idx)
        return int(cid[2] != '1' and qtr < 6)

    def evaluate_fitness(self, chrom: SchedChrom1d):
        """Currently, we only have hard constraints. Therefore, maximum fitness is 0.

        Args:
            chrom: Chromosome whose fitness is being processed.

        Returns:
            Evaluated fitness value. This also updates `chrom.fitness`.

        The constraints that are currently encoded are:
            HC # 1 - units per quarter MUST NOT exceed 18
            HC # 2 - core courses MUST BE taken
            HC # 3 - courses MUST BE offered in the assigned quarter
            HC # 4 - prerequisites MUST BE satisfied
            HC # 5 - track requirements MUST BE satisfied

            SC # 1 - number of courses per quarter SHOULD NOT exceed 3
            SC # 2 - core courses SHOULD BE taken earlier on
            SC # 3 - 200/300 level courses SHOUlD NOT be taken too early

        Hard Constraints (HC) are associated with higher penalties, and Soft Constraints with lower.
        First, iterate through each gene of the Chromosome. Evaluate all local constraint violations.
        Collect information from local search procedure. Finally, evaluate all global constraint violations.

        Note that the state MUST BE cleared after each evaluation.
        """
        penalty = 0

        for idx in range(len(chrom.data)):
            if chrom.data[idx] < 0:
                continue

            for constraint in self._loc_cstrt_idx:
                count = self._loc_cstrt_fns[constraint](chrom, idx)
                penalty += count * self._loc_cstrt_pen[constraint]

            for collector in self._collectors:
                collector(chrom, idx)

        for constraint in self._glob_cstrt_idx:
            count = self._glob_cstrt_fns[constraint](chrom)
            penalty += count * self._glob_cstrt_pen[constraint]

        self._states.clear()

        chrom.fitness = -penalty
        return -penalty

    def pretty_print(self, chrom: SchedChrom1d):
        d = chrom.destructure()
        for i in range(-1, 12):
            print(f'Quarter {i} ==> {[self._to_cid(j) for j in d.get(i, [])]}')
