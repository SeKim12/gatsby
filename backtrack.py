# import collections
# import copy
# from typing import Dict, List
#
# from courseUtil import CourseUtil, CourseBulletin
#
# class CSP:
#     def __init__(self):
#         # Total number of variables in the CSP.
#         self.numVars = 0
#
#         # The list of variable names in the same order as they are added. A
#         # variable name can be any hashable objects, for example: int, str,
#         # or any tuple with hashtable objects.
#         self.variables = []
#
#         # Each key K in this dictionary is a variable name.
#         # values[K] is the list of domain values that variable K can take on.
#         self.values = {}
#
#         # Each entry is a unary factor table for the corresponding variable.
#         # The factor table corresponds to the weight distribution of a variable
#         # for all added unary factor functions. If there's no unary function for
#         # a variable K, there will be no entry for K in unaryFactors.
#         # E.g. if B \in ['a', 'b'] is a variable, and we added two
#         # unary factor functions f1, f2 for B,
#         # then unaryFactors[B]['a'] == f1('a') * f2('a')
#         self.unaryFactors = {}
#
#         # Each entry is a dictionary keyed by the name of the other variable
#         # involved. The value is a binary factor table, where each table
#         # stores the factor value for all possible combinations of
#         # the domains of the two variables for all added binary factor # functions.
#         # The table is represented as a dictionary of dictionary.
#         # As an example, if we only have two variables
#         # A \in ['b', 'c'],  B \in ['a', 'b']
#         # and we've added two binary functions f1(A,B) and f2(A,B) to the CSP,
#         # then binaryFactors[A][B]['b']['a'] == f1('b','a') * f2('b','a').
#         # binaryFactors[A][A] should return a key error since a variable
#         # shouldn't have a binary factor table with itself.
#
#         self.binaryFactors = {}
#
#     def add_variable(self, var, domain: List) -> None:
#         """
#         Add a new variable to the CSP.
#         """
#         if var in self.variables:
#             raise Exception("Variable name already exists: %s" % str(var))
#
#         self.numVars += 1
#         self.variables.append(var)
#         self.values[var] = domain
#         self.unaryFactors[var] = None
#         self.binaryFactors[var] = dict()
#
#     def get_neighbor_vars(self, var) -> List:
#         """
#         Returns a list of variables which are neighbors of |var|.
#         """
#         return list(self.binaryFactors[var].keys())
#
#     def add_unary_factor(self, var, factorFunc) -> None:
#         """
#         Add a unary factor function for a variable. Its factor
#         value across the domain will be *merged* with any previously added
#         unary factor functions through elementwise multiplication.
#
#         How to get unary factor value given a variable |var| and
#         value from the domain |val|?
#         => csp.unaryFactors[var][val]
#         """
#         factor = {val:float(factorFunc(val)) for val in self.values[var]}
#         if self.unaryFactors[var] is not None:
#             assert len(self.unaryFactors[var]) == len(factor)
#             self.unaryFactors[var] = {val:self.unaryFactors[var][val] * \
#                 factor[val] for val in factor}
#         else:
#             self.unaryFactors[var] = factor
#
#     def add_binary_factor(self, var1, var2, factor_func):
#         """
#         Takes two variable names and a binary factor function
#         |factorFunc|, add to binaryFactors. If the two variables already
#         had binaryFactors added earlier, they will be *merged* through element
#         wise multiplication.
#
#         How to get binary factor value given a variable |var1| with value |val1|
#         and variable |var2| with value |val2|?
#         => csp.binaryFactors[var1][var2][val1][val2]
#         """
#         # never shall a binary factor be added over a single variable
#         try:
#             assert var1 != var2
#         except:
#             print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#             print('!! Tip:                                                                      !!')
#             print('!! You are adding a binary factor over a same variable...                    !!')
#             print('!! Please check your code and avoid doing this.                              !!')
#             print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#             raise
#
#         self.update_binary_factor_table(var1, var2,
#             {val1: {val2: float(factor_func(val1, val2)) \
#                 for val2 in self.values[var2]} for val1 in self.values[var1]})
#         self.update_binary_factor_table(var2, var1, \
#             {val2: {val1: float(factor_func(val1, val2)) \
#                 for val1 in self.values[var1]} for val2 in self.values[var2]})
#
#     def update_binary_factor_table(self, var1, var2, table):
#         """
#         Private method you can skip for 0c, might be useful for 1c though.
#         Update the binary factor table for binaryFactors[var1][var2].
#         If it exists, element-wise multiplications will be performed to merge
#         them together.
#         """
#         if var2 not in self.binaryFactors[var1]:
#             self.binaryFactors[var1][var2] = table
#         else:
#             currentTable = self.binaryFactors[var1][var2]
#             for i in table:
#                 for j in table[i]:
#                     assert i in currentTable and j in currentTable[i]
#                     currentTable[i][j] *= table[i][j]
#
#
# class BacktrackingSearch:
#     def __init__(self):
#         self.optimalAssignment = {}
#         self.optimalWeight = 0
#
#         # Keep track of the number of optimal assignments and assignments. These
#         # two values should be identical when the CSP is unweighted or only has binary
#         # weights.
#         self.numOptimalAssignments = 0
#         self.numAssignments = 0
#
#         # Keep track of the number of times backtrack() gets called.
#         self.numOperations = 0
#
#         # Keep track of the number of operations to get to the very first successful
#         # assignment (doesn't have to be optimal).
#         self.firstAssignmentNumOperations = 0
#
#         # List of all solutions found.
#         self.allAssignments = []
#         self.allOptimalAssignments = []
#     #
#
#     def reset_results(self) -> None:
#         """
#         This function resets the statistics of the different aspects of the
#         CSP solver. We will be using the values here for grading, so please
#         do not make any modification to these variables.
#         """
#         # Keep track of the best assignment and weight found.
#         self.optimalAssignment = {}
#         self.optimalWeight = 0
#
#         # Keep track of the number of optimal assignments and assignments. These
#         # two values should be identical when the CSP is unweighted or only has binary
#         # weights.
#         self.numOptimalAssignments = 0
#         self.numAssignments = 0
#
#         # Keep track of the number of times backtrack() gets called.
#         self.numOperations = 0
#
#         # Keep track of the number of operations to get to the very first successful
#         # assignment (doesn't have to be optimal).
#         self.firstAssignmentNumOperations = 0
#
#         # List of all solutions found.
#         self.allAssignments = []
#         self.allOptimalAssignments = []
#     #
#     def print_stats(self) -> None:
#         """
#         Prints a message summarizing the outcome of the solver.
#         """
#         if self.optimalAssignment:
#             print(f'Found {self.numOptimalAssignments} optimal assignments \
#                     with weight {self.optimalWeight} in {self.numOperations} operations')
#             print(
#                 f'First assignment took {self.firstAssignmentNumOperations} operations')
#         else:
#             print(
#                 "No consistent assignment to the CSP was found. The CSP is not solvable.")
#
#     def get_delta_weight(self, assignment: Dict, var, val) -> float:
#         """
#         Given a CSP, a partial assignment, and a proposed new value for a variable,
#         return the change of weights after assigning the variable with the proposed
#         value.
#
#         @param assignment: A dictionary of current assignment. Unassigned variables
#             do not have entries, while an assigned variable has the assigned value
#             as value in dictionary. e.g. if the domain of the variable A is [5,6],
#             and 6 was assigned to it, then assignment[A] == 6.
#         @param var: name of an unassigned variable.
#         @param val: the proposed value.
#
#         @return w: Change in weights as a result of the proposed assignment. This
#             will be used as a multiplier on the current weight.
#         """
#         assert var not in assignment
#         w = 1.0
#         if self.csp.unaryFactors[var]:
#             w *= self.csp.unaryFactors[var][val]
#             if w == 0:
#                 return w
#         for var2, factor in list(self.csp.binaryFactors[var].items()):
#             if var2 not in assignment:
#                 continue  # Not assigned yet
#             w *= factor[val][assignment[var2]]
#             if w == 0:
#                 return w
#         return w
#
#     def satisfies_constraints(self, assignment: Dict, var, val) -> bool:
#         """
#         Given a CSP, a partial assignment, and a proposed new value for a variable,
#         return whether or not assigning the variable with the proposed new value
#         still statisfies all of the constraints.
#
#         @param assignment: A dictionary of current assignment. Unassigned variables
#             do not have entries, while an assigned variable has the assigned value
#             as value in dictionary. e.g. if the domain of the variable A is [5,6],
#             and 6 was assigned to it, then assignment[A] == 6.
#         @param var: name of an unassigned variable.
#         @param val: the proposed value.
#
#         @return w: Change in weights as a result of the proposed assignment. This
#             will be used as a multiplier on the current weight.
#         """
#         return self.get_delta_weight(assignment, var, val) != 0
#
#     def solve(self, csp: CSP, mcv: bool = False, ac3: bool = False) -> None:
#         """
#         Solves the given weighted CSP using heuristics as specified in the
#         parameter. Note that unlike a typical unweighted CSP where the search
#         terminates when one solution is found, we want this function to find
#         all possible assignments. The results are stored in the variables
#         described in reset_result().
#
#         @param csp: A weighted CSP.
#         @param mcv: When enabled, Most Constrained Variable heuristics is used.
#         @param ac3: When enabled, AC-3 will be used after each assignment of an
#             variable is made.
#         """
#         # CSP to be solved.
#         self.csp = csp
#
#         # Set the search heuristics requested asked.
#         self.mcv = mcv
#         self.ac3 = ac3
#
#         # Reset solutions from previous search.
#         self.reset_results()
#
#         # The dictionary of domains of every variable in the CSP.
#         self.domains = {
#             var: list(self.csp.values[var]) for var in self.csp.variables}
#
#         # Perform backtracking search.
#         self.backtrack({}, 0, 1)
#         # Print summary of solutions.
#         self.print_stats()
#
#     def backtrack(self, assignment: Dict, numAssigned: int, weight: float) -> None:
#         """
#         Perform the back-tracking algorithms to find all possible solutions to
#         the CSP.
#
#         @param assignment: A dictionary of current assignment. Unassigned variables
#             do not have entries, while an assigned variable has the assigned value
#             as value in dictionary. e.g. if the domain of the variable A is [5,6],
#             and 6 was assigned to it, then assignment[A] == 6.
#         @param numAssigned: Number of currently assigned variables
#         @param weight: The weight of the current partial assignment.
#         """
#
#         self.numOperations += 1
#         assert weight > 0
#         if numAssigned == self.csp.numVars:
#             # A satisfiable solution have been found. Update the statistics.
#             self.numAssignments += 1
#             newAssignment = {}
#             for var in self.csp.variables:
#                 newAssignment[var] = assignment[var]
#             self.allAssignments.append(newAssignment)
#             return
#             #
#             # if len(self.optimalAssignment) == 0 or weight >= self.optimalWeight:
#             #     if weight == self.optimalWeight:
#             #         self.numOptimalAssignments += 1
#             #         self.allOptimalAssignments.append(newAssignment)
#             #     else:
#             #         self.numOptimalAssignments = 1
#             #         self.allOptimalAssignments = [newAssignment]
#             #     self.optimalWeight = weight
#             #
#             #     self.optimalAssignment = newAssignment
#             #     if self.firstAssignmentNumOperations == 0:
#             #         self.firstAssignmentNumOperations = self.numOperations
#             # return
#
#         # Select the next variable to be assigned.
#         var = self.get_unassigned_variable(assignment)
#         # Get an ordering of the values.
#         ordered_values = self.domains[var]
#
#         # Continue the backtracking recursion using |var| and |ordered_values|.
#         if not self.ac3:
#             # When arc consistency check is not enabled.
#             for val in ordered_values:
#                 deltaWeight = self.get_delta_weight(assignment, var, val)
#                 if deltaWeight > 0:
#                     assignment[var] = val
#                     self.backtrack(assignment, numAssigned +
#                                    1, weight * deltaWeight)
#                     del assignment[var]
#         else:
#             # Arc consistency check is enabled. This is helpful to speed up 3c.
#             for val in ordered_values:
#                 deltaWeight = self.get_delta_weight(assignment, var, val)
#                 if deltaWeight > 0:
#                     assignment[var] = val
#                     # create a deep copy of domains as we are going to look
#                     # ahead and change domain values
#                     localCopy = copy.deepcopy(self.domains)
#                     # fix value for the selected variable so that hopefully we
#                     # can eliminate values for other variables
#                     self.domains[var] = [val]
#
#                     # enforce arc consistency
#                     self.apply_arc_consistency(var)
#
#                     self.backtrack(assignment, numAssigned +
#                                    1, weight * deltaWeight)
#                     # restore the previous domains
#                     self.domains = localCopy
#                     del assignment[var]
#
#     def get_unassigned_variable(self, assignment: Dict):
#         """
#         Given a partial assignment, return a currently unassigned variable.
#
#         @param assignment: A dictionary of current assignment. This is the same as
#             what you've seen so far.
#
#         @return var: a currently unassigned variable. The type of the variable
#             depends on what was added with csp.add_variable
#         """
#
#         if not self.mcv:
#             # Select a variable without any heuristics.
#             for var in self.csp.variables:
#                 if var not in assignment:
#                     return var
#         else:
#             # Problem 1b
#             # Heuristic: most constrained variable (MCV)
#             # Select a variable with the least number of remaining domain values.
#             # Hint: given var, self.domains[var] gives you all the possible values.
#             #       Make sure you're finding the domain of the right variable!
#             # Hint: satisfies_constraints determines whether or not assigning a
#             #       variable to some value given a partial assignment continues
#             #       to satisfy all constraints.
#             # Hint: for ties, choose the variable with lowest index in self.csp.variables
#             # BEGIN_YOUR_CODE (our solution is 13 lines of code, but don't worry if you deviate from this)
#             min_count = float('inf')
#             most_constrained = None
#             for var in self.csp.variables:
#                 if var not in assignment:
#                     candidates = self.domains[var]
#                     count = 0
#                     for val in candidates:
#                         if self.satisfies_constraints(assignment, var, val):
#                             count += 1
#                     if count < min_count:
#                         min_count = count
#                         most_constrained = var
#             return most_constrained
#             # END_YOUR_CODE
#
#     def apply_arc_consistency(self, var) -> None:
#         """
#         Perform the AC-3 algorithm. The goal is to reduce the size of the
#         domain values for the unassigned variables based on arc consistency.
#
#         @param var: The variable whose value has just been set.
#         """
#
#         def remove_inconsistent_values(var1, var2):
#             removed = False
#             # the binary factor must exist because we add var1 from var2's neighbor
#             factor = self.csp.binaryFactors[var1][var2]
#             for val1 in list(self.domains[var1]):
#                 # Note: in our implementation, it's actually unnecessary to check unary factors,
#                 #       because in get_delta_weight() unary factors are always checked.
#                 if (self.csp.unaryFactors[var1] and self.csp.unaryFactors[var1][val1] == 0) or \
#                         all(factor[val1][val2] == 0 for val2 in self.domains[var2]):
#                     self.domains[var1].remove(val1)
#                     removed = True
#             return removed
#
#         queue = [var]
#         while len(queue) > 0:
#             curr = queue.pop(0)
#             for neighbor in self.csp.get_neighbor_vars(curr):
#                 if remove_inconsistent_values(neighbor, curr):
#                     queue.append(neighbor)
#
#
# def create_sum_variable(csp: CSP, name: str, variables: List, maxSum: int) -> tuple:
#     """
#     Given a list of |variables| each with non-negative integer domains,
#     returns the name of a new variable with domain range(0, maxSum+1), such that
#     it's consistent with the value |n| iff the assignments for |variables|
#     sums to |n|.
#
#     @param name: Prefix of all the variables that are going to be added.
#         Can be any hashable objects. For every variable |var| added in this
#         function, it's recommended to use a naming strategy such as
#         ('sum', |name|, |var|) to avoid conflicts with other variable names.
#     @param variables: A list of variables that are already in the CSP that
#         have non-negative integer values as its domain.
#     @param maxSum: An integer indicating the maximum sum value allowed. You
#         can use it to get the auxiliary variables' domain
#
#     @return result: The name of a newly created variable with domain range
#         [0, maxSum] such that it's consistent with an assignment of |n|
#         iff the assignment of |variables| sums to |n|.
#     """
#
#     result = ('sum', name, 'aggregated')
#     csp.add_variable(result, list(range(maxSum + 1)))
#
#     if len(variables) == 0:
#         csp.add_unary_factor(result, lambda x: x == 0)
#         return result
#
#     domain = []
#     for i in range(maxSum + 1):
#         for j in range(i, maxSum + 1):
#             domain.append((i, j))
#
#     for i in range(len(variables)):
#         csp.add_variable(('sum', name, str(i)), domain)
#
#     csp.add_unary_factor(('sum', name, '0'), lambda x: x[0] == 0)
#
#     for i in range(len(variables)):
#         f = ('sum', name, str(i))
#         csp.add_binary_factor(f, variables[i], lambda x, y: x[1] == x[0] + y)
#
#     for i in range(len(variables) - 1):
#         f0 = ('sum', name, str(i))
#         f1 = ('sum', name, str(i + 1))
#         csp.add_binary_factor(f0, f1, lambda x, y: x[1] == y[0])
#
#     csp.add_binary_factor(
#         ('sum', name, str(len(variables) - 1)), result, lambda x, y: x[1] == y)
#
#     return result
#
# ############################################################
# # Problem 2
#
# # A class providing methods to generate CSP that can solve the course scheduling
# # problem.
#
#
# class SchedulingCSPConstructor:
#     def __init__(self, bulletin: CourseUtil):
#         """
#         Saves the necessary data.
#
#         @param bulletin: Stanford Bulletin that provides a list of courses
#         @param profile: A student's profile and requests
#         """
#
#         self.bulletin = bulletin
#         self.to_qid = lambda q: "Aut" if q % 3 == 0 else "Win" if q % 3 == 1 else "Spr"
#
#     def add_variables(self, csp: CSP) -> None:
#         """
#         Adding the variables into the CSP. Each variable, (request, quarter),
#         can take on the value of one of the courses requested in request or None.
#         For instance, for quarter='Aut2013', and a request object, request, generated
#         from 'CS221 or CS246', (request, quarter) should have the domain values
#         ['CS221', 'CS246', None]. Conceptually, if var is assigned 'CS221'
#         then it means we are taking 'CS221' in 'Aut2013'. If it's None, then
#         we are not taking either of them in 'Aut2013'.
#
#         @param csp: The CSP where the additional constraints will be added to.
#         """
#
#         for course in self.bulletin.get_courses():
#             csp.add_variable(course, list(range(-1, 12)))
#
#     def add_bulletin_constraints(self, csp: CSP) -> None:
#         """
#         Add the constraints that a course can only be taken if it's offered in
#         that quarter.
#
#         @param csp: The CSP where the additional constraints will be added to.
#         """
#         for course in self.bulletin.get_courses():
#             csp.add_unary_factor(course, lambda q: self.to_qid(q) in self.bulletin.get_quarters_offered(course))
#
#     # def add_norepeating_constraints(self, csp: CSP) -> None:
#     #     """
#     #     No course can be repeated. Coupling with our problem's constraint that
#     #     only one of a group of requested course can be taken, this implies that
#     #     every request can only be satisfied in at most one quarter.
#     #
#     #     @param csp: The CSP where the additional constraints will be added to.
#     #     """
#     #
#     #     for request in self.profile.requests:
#     #         for quarter1 in self.profile.quarters:
#     #             for quarter2 in self.profile.quarters:
#     #                 if quarter1 == quarter2:
#     #                     continue
#     #                 csp.add_binary_factor((request, quarter1), (request, quarter2),
#     #                                       lambda cid1, cid2: cid1 is None or cid2 is None)
#
#     def get_basic_csp(self) -> CSP:
#         """
#         Return a CSP that only enforces the basic constraints that a course can
#         only be taken when it's offered and that a request can only be satisfied
#         in at most one quarter.
#
#         @return csp: A CSP where basic variables and constraints are added.
#         """
#
#         csp = CSP()
#         self.add_variables(csp)
#         self.add_bulletin_constraints(csp)
#         return csp
#     #
#     # def add_quarter_constraints(self, csp: CSP) -> None:
#     #     """
#     #     If the profile explicitly wants a request to be satisfied in some given
#     #     quarters, e.g. Aut2013, then add constraints to not allow that request to
#     #     be satisfied in any other quarter. If a request doesn't specify the
#     #     quarter(s), do nothing.
#     #
#     #     @param csp: The CSP where the additional constraints will be added to.
#     #     """
#     #     # Problem 2a
#     #     # Hint: If a request doesn't specify the quarter(s), do nothing.
#     #     # Hint: To check which quarters are specified by a request variable
#     #     #       named `request`, use request.quarters (NOT self.profile.quarters).
#     #     # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
#     #     for request in self.profile.requests:
#     #         if not request.quarters:
#     #             continue
#     #         for quarter in self.profile.quarters:
#     #             csp.add_unary_factor((request, quarter), lambda cid: cid is None or quarter in request.quarters)
#     #     # END_YOUR_CODE
#     #
#     # def add_request_weights(self, csp: CSP) -> None:
#     #     """
#     #     Incorporate weights into the CSP. By default, a request has a weight
#     #     value of 1 (already configured in Request). You should only use the
#     #     weight when one of the requested course is in the solution. A
#     #     unsatisfied request should also have a weight value of 1.
#     #
#     #     @param csp: The CSP where the additional constraints will be added to.
#     #     """
#     #
#     #     for request in self.profile.requests:
#     #         for quarter in self.profile.quarters:
#     #             csp.add_unary_factor((request, quarter),
#     #                                  lambda cid: request.weight if cid != None else 1.0)
#
#     def add_prereq_constraints(self, csp: CSP) -> None:
#         """
#         Adding constraints to enforce prerequisite. A course can have multiple
#         prerequisites. You can assume that *all courses in req.prereqs are
#         being requested*. Note that if our parser inferred that one of your
#         requested course has additional prerequisites that are also being
#         requested, these courses will be added to req.prereqs. You will be notified
#         with a message when this happens. Also note that req.prereqs apply to every
#         single course in req.cids. If a course C has prerequisite A that is requested
#         together with another course B (i.e. a request of 'A or B'), then taking B does
#         not count as satisfying the prerequisite of C. You cannot take a course
#         in a quarter unless all of its prerequisites have been taken *before* that
#         quarter. You should take advantage of get_or_variable().
#
#         @param csp: The CSP where the additional constraints will be added to.
#         """
#         for course in self.bulletin.get_courses():
#             for prereq in self.bulletin.get_prereqs(course):
#                 if prereq in csp.variables:
#                     csp.add_binary_factor(course, prereq, lambda q1, q2: q2 != -1 and q2 <= q1)
#
#     # def add_unit_constraints(self, csp: CSP) -> None:
#     #     """
#     #     Add constraints to the CSP to ensure that course units are correctly assigned.
#     #     This means meeting two conditions:
#     #
#     #     1- If a course is taken in a given quarter, it should be taken for
#     #        a number of units that is within bulletin.courses[cid].minUnits/maxUnits.
#     #        If not taken, it should be 0.
#     #     2- In each quarter, the total number of units of courses taken should be between
#     #        profile.minUnits/maxUnits, inclusively.
#     #        You should take advantage of create_sum_variable() to implement this.
#     #
#     #     For every requested course, you must create a variable named (courseId, quarter)
#     #     (e.g. ('CS221', 'Aut2013')) and its assigned value is the number of units.
#     #     This variable is how our solution extractor will obtain the number of units,
#     #     so be sure to add it.
#     #
#     #     For a request 'A or B', if you choose to take A, then you must use a unit
#     #     number that's within the range of A.
#     #
#     #     @param csp: The CSP where the additional constraints will be added to.
#     #     """
#     #     # Problem 2b
#     #     # Hint 1: read the documentation above carefully
#     #     # Hint 2: the domain for each (courseId, quarter) variable should contain 0
#     #     #         because the course might not be taken
#     #     # Hint 3: add appropriate binary factor between (request, quarter) and
#     #     #         (courseId, quarter) variables. Remember that a course can only
#     #     #         be requested at most once in a profile and that if you have a
#     #     #         request such as 'request A or B', then A being taken must
#     #     #         mean that B is NOT taken in the schedule.
#     #     # Hint 4: you must ensure that the sum of units per quarter for your schedule
#     #     #         are within the min and max threshold inclusive
#     #     # Hint 5: use nested functions and lambdas like what get_or_variable and
#     #     #         add_prereq_constraints do
#     #     # Hint 6: don't worry about quarter constraints in each Request as they'll
#     #     #         be enforced by the constraints added by add_quarter_constraints
#     #     # Hint 7: if 'a' and 'b' are the names of two variables constrained to have
#     #     #         sum 7, you can create a variable for their sum
#     #     #         by calling var = create_sum_variable(csp, SOME_NAME, ['a', 'b'], 7).
#     #     #         Note: 7 in the call just helps create the domain; it does
#     #     #         not define a constraint. You can add a constraint to the
#     #     #         sum variable (e.g., to make sure that the number of units
#     #     #         taken in a quarter are within a specific range) just as
#     #     #         you've done for other variables.
#     #     # Hint 8: if having trouble with starting, here is some general structure
#     #     #         that you could use to get started:
#     #     #           loop through self.profile.quarters
#     #     #               ...
#     #     #               loop through self.profile.requests
#     #     #                   loop through request.cids
#     #     #                       ...
#     #     #               ...
#     #     # BEGIN_YOUR_CODE (our solution is 21 lines of code, but don't worry if you deviate from this)
#     #     # def ff(cid_, units_):
#     #     #     print(cid_, units_)
#     #     #     print(cid, units_)
#     #     #     if cid_ == cid:
#     #     #         print(units_ > 0)
#     #     #         return units_ > 0
#     #     #     else:
#     #     #         print(units_ == 0)
#     #     #         return units_ == 0
#     #     for i in range(12):
#     #         courses = []
#     #         for course in self.bulletin.get_courses():
#     #
#     #     for quarter in self.profile.quarters:
#     #         courses = []
#     #         for request in self.profile.requests:
#     #             for cid in request.cids:
#     #                 course = self.bulletin.courses[cid]
#     #                 domain = [0] + list(range(course.minUnits, course.maxUnits + 1))
#     #                 courses.append((cid, quarter))
#     #                 csp.add_variable((cid, quarter), domain)
#     #                 csp.add_binary_factor((request, quarter), (cid, quarter),
#     #                                       lambda cur_cid, units: units > 0 if cur_cid == cid else units == 0)
#     #         sum_var = create_sum_variable(csp, quarter, courses, self.profile.maxUnits)
#     #         csp.add_unary_factor(sum_var, lambda x: self.profile.maxUnits >= x >= self.profile.minUnits)
#
#
#
#         # raise Exception("Not implemented yet")
#         # END_YOUR_CODE
#
#     def add_all_additional_constraints(self, csp: CSP) -> None:
#         """
#         Add all additional constraints to the CSP.
#
#         @param csp: The CSP where the additional constraints will be added to.
#         """
#         # self.add_quarter_constraints(csp)
#         # self.add_request_weights(csp)
#         self.add_prereq_constraints(csp)
#         # self.add_unit_constraints(csp)
#
#
# if __name__ == '__main__':
#     bulletin = CourseBulletin('courses.json')
#     cspConst = SchedulingCSPConstructor(bulletin)
#     csp = cspConst.get_basic_csp()
#     cspConst.add_all_additional_constraints(csp)
#     alg = BacktrackingSearch()
#     alg.solve(csp, mcv=True, ac3 = True)
#     # ga = GeneticAlgorithm(120, 700, cop, Selection.rank_selection, Crossover.uniform_crossover,
#     #                       Mutation.shuffle_mutate, Pc=0.8, Pm=0.08, max_fitness=0, tabu=True, verbose=False)
#     # ga.run()
#     #
#     # # ga.plot_fitness()
#     # # profile = util.Profile(bulletin, 'profile2b3.txt')
#     # # cspConstructor = submission.SchedulingCSPConstructor(bulletin, copy.deepcopy(profile))
#     # csp = cspConstructor.get_basic_csp()
#     # cspConstructor.add_all_additional_constraints(csp)
#     # alg = submission.BacktrackingSearch()
#     # alg.solve(csp, mcv = True, ac3 = True)