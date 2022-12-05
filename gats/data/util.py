"""Utility class and functions to work with course data.

The `CourseParser` interface decouples data collection/processing and usage in COP.
The `CourseBulletin` class and its associated methods are implementations specific to
`model.scheduler.Scheduler,` and is only intended to work with `data/sample_courses.json`
"""

import json
import re
from typing import List, Dict, Any, Set
from abc import ABC, abstractmethod


class CourseParser(ABC):
    """
    Any class dealing with external course data must satisfy this interface.
    """

    @abstractmethod
    def get_courses(self) -> List[Any]:
        pass

    @abstractmethod
    def get_num_courses(self) -> int:
        pass

    @abstractmethod
    def get_quarters_offered(self, uid: str) -> List[Any]:
        pass

    @abstractmethod
    def get_prereqs(self, uid: str) -> List[Any]:
        pass

    @abstractmethod
    def get_max_units(self, uid: str) -> int:
        pass


class Course:
    """
    Course Object from CS221 Scheduling Homework
    """

    def __init__(self, info: Dict):
        self.__dict__.update(info)

    # Return whether this course is offered in |quarter| (e.g., Aut2013).
    def is_offered_in(self, quarter: str) -> bool:
        return any(quarter.startswith(q) for q in self.quarters)

    def short_str(self) -> str:
        return f"{self.cid}: {self.name}"

    def __str__(self):
        return f"Course: {self.cid}, name: {self.name}, quarters: {self.quarters}, \
                units: {self.minUnits}-{self.maxUnits}, prereqs: {self.prereqs}"


class CourseBulletin(CourseParser):
    """
    Course Util from CS221 Scheduling Homework
    Reads ONLY CS course data from `sample_courses.json`
    """

    def __init__(self, coursesPath: str):
        self.courses = {}
        self.course_structs = []
        info = json.loads(open(coursesPath).read())
        for courseInfo in list(info.values()):
            if re.match(r"^CS\d", courseInfo["cid"]):
                course = Course(courseInfo)
                self.courses[course.cid] = course

    def get_num_courses(self):
        return len(self.courses)

    def get_quarters_offered(self, cid: str):
        return self.courses[cid].quarters

    def get_prereqs(self, cid: str):
        return self.courses[cid].prereqs

    def get_courses(self):
        return list(self.courses.keys())

    def get_max_units(self, cid: str):
        return self.courses[cid].maxUnits


def load_core() -> Set[str]:
    return {"CS103", "CS109", "CS106A", "CS106B", "CS107", "CS110", "CS161"}


def load_track(track: str) -> Dict[str, Set[str]]:
    reqs = dict()
    if track == "AI":
        parts = ["A", "B1", "B2", "C"]
        list_A = ["CS221"]
        list_B1 = ["CS224R", "CS228", "CS229", "CS229M", "CS229T", "CS234", "CS238"]
        list_B2 = ["CS124", "CS224N", "CS224S", "CS224U", "CS224V"]
        list_C = [
            "CS157",
            "CS205L",
            "CS230",
            "CS236",
            "CS257",
            "CS235",
            "CS279",
            "CS371",
        ]
        for part in parts:
            reqs[part] = set()
            p_list = eval("list_" + part)
            for course in p_list:
                reqs[part].add(course)
    if track == "HCI":
        parts = ["A1", "A2", "A3", "B", "C"]
        list_A1 = ["CS147"]
        list_A2 = ["CS247"]
        list_A3 = ["CS347"]
        list_B = ["CS143"]
        list_C = ["CS278", "CS448B"]
        for part in parts:
            reqs[part] = set()
            p_list = eval("list_" + part)
            for course in p_list:
                reqs[part].add(course)
    if track == "Systems":
        parts = ["A", "B", "C"]
        list_A = ["CS140"]
        list_B = ["CS143"]
        list_C = ["CS241", "CS269Q", "CS316", "CS341", "CS344"]
        for part in parts:
            reqs[part] = set()
            p_list = eval("list_" + part)
            for course in p_list:
                reqs[part].add(course)
    if track == "Theory":
        parts = ["A", "B", "C1", "C2"]
        list_A = ["CS154"]
        list_B = ["CS168", "CS255", "CS261", "CS265", "CS268"]
        list_C1 = ["CS143", "CS151", "CS155", "CS157", "CS163", "CS166"]
        list_C2 = [
            "CS205L",
            "CS228",
            "CS233",
            "CS235",
            "CS236",
            "CS242",
            "CS250",
            "CS251",
            "CS252",
            "CS254",
            "CS259",
        ]
        for part in parts:
            reqs[part] = set()
            p_list = eval("list_" + part)
            for course in p_list:
                reqs[part].add(course)
    # add senior project at the end
    sp = ["CS191", "CS194", "CS210B", "CS294"]
    reqs["S"] = set()
    for course in sp:
        reqs["S"].add(course)
    return reqs
