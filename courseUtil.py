from typing import List, Dict, Any
import json
import re
from abc import ABC, abstractmethod


class CourseUtil(ABC):
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

    def short_str(self) -> str: return f'{self.cid}: {self.name}'

    def __str__(self):
        return f'Course: {self.cid}, name: {self.name}, quarters: {self.quarters}, \
                units: {self.minUnits}-{self.maxUnits}, prereqs: {self.prereqs}'


class CourseBulletin(CourseUtil):
    """
    Course Util from CS221 Scheduling Homework
    Reads ONLY CS course data from `courses.json`
    """
    def __init__(self, coursesPath: str):
        self.courses = {}
        info = json.loads(open(coursesPath).read())
        for courseInfo in list(info.values()):
            if re.match(r'^CS\d', courseInfo["cid"]):
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