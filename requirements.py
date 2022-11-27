import numpy as np
import numpy.typing as npt

from typing import List, Any, Tuple, Iterator
from genetic import COP, Chromosome, TdChromosome

class Requirements:
    @staticmethod
    def load_core():
        return ["CS103", "CS109", "CS106A", "CS106B", "CS107", "CS110", "CS161"]
    
    @staticmethod
    def load_track(cop: COP):
        reqs = dict()
        if cop.track == "AI":
            parts = ["A", "B1", "B2", "C"]
            list_A = ["CS221"]
            list_B1 = ["CS224R", "CS228", "CS229", "CS229M", "CS229T", "CS234", "CS238"]
            list_B2 = ["CS124", "CS224N", "CS224S", "CS224U", "CS224V"]
            list_C = ["CS157", "CS205L", "CS230", "CS236", "CS257", "CS235", "CS279", "CS371"]
            for part in parts:
                reqs[part] = set()
                p_list = eval("list_" + part)
                for course in p_list:
                    reqs[part].add(course)
        if cop.track == "HCI":
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
        if cop.track == "Systems":
            parts = ["A", "B", "C"]
            list_A = ["CS140"]
            list_B = ["CS143"]
            list_C = ["CS241", "CS269Q", "CS316", "CS341", "CS344"]
            for part in parts:
                reqs[part] = set()
                p_list = eval("list_" + part)
                for course in p_list:
                    reqs[part].add(course)
        if cop.track == "Theory":
            parts = ["A", "B", "C1", "C2"]
            list_A = ["CS154"]
            list_B = ["CS168", "CS255", "CS261", "CS265", "CS268"]
            list_C1 = ["CS143", "CS151", "CS155", "CS157", "CS163", "CS166"]
            list_C2 = ["CS205L", "CS228", "CS233", "CS235", "CS236", "CS242", "CS250", "CS251", "CS252", "CS254", "CS259"]
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