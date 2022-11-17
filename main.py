# import numpy as np
# from nqueens import NQueens
#
# Pc = 0.7
# Pm = 0.05
# Pf = 0.4
#
#
# def generate_examples(n: int):
#     res = []
#     for i in range(n):
#         res.append(np.random.random_integers(0, 4, (4, 4)))
#     return res
#
#
# def stochastic_rank(population):
#     ranked = population
#     for i in range(len(ranked)):
#         swapped = False
#         for j in range(len(ranked) - 1):
#             c1, c2 = ranked[j], ranked[j + 1]
#             pen1 = NQueens.evaluate_penalty(c1)
#             pen2 = NQueens.evaluate_penalty(c2)
#             if (pen1 == 0 and pen2 == 0) or np.random.random() < Pf:
#                 if NQueens.evaluate_fitness(c1) > NQueens.evaluate_fitness(c2):
#                     ranked[j], ranked[j + 1] = ranked[j + 1], ranked[j]
#                     swapped = True
#             else:
#                 if pen1 < pen2:
#                     ranked[j], ranked[j + 1] = ranked[j + 1], ranked[j]
#                     swapped = True
#
#         if not swapped:
#             break
#     return ranked
#
#
# def mutate(c):
#     """
#     Two-dimensional Two-point Swapping Mutation as described in https://downloads.hindawi.com/journals/mpe/2015/906305.pdf
#     :param c: individual chromosome to be mutated
#     :return: None (in-place)
#     """
#     if np.random.random() < Pm:
#         rr = np.random.randint(c.shape[0])
#         rc = np.random.randint(c.shape[1])
#         rrp, rcp = rr, rc
#         while rrp == rr and rcp == rc:
#             rrp = np.random.randint(c.shape[0])
#             rcp = np.random.randint(c.shape[1])
#
#         c[rr, rc], c[rrp, rcp] = c[rrp, rcp], c[rr, rc]
#
#
# def crossover(cp1, cp2):
#     """
#     Two-dimensional Substring Crossover as described in https://downloads.hindawi.com/journals/mpe/2015/906305.pdf
#     Randomly sample row crossover point (R_r) and column crossover point (R_c)
#     With 0.5 probability, perform horizontal crossover or vertical crossover
#     :param cp1: Crossover Parent1
#     :param cp2: Crossover Parent2
#     :return: Crossover Offspring1, Crossover Offspring2
#     """
#     def horizontal_crossover():
#         co1 = np.zeros(cp1.shape)
#         co2 = np.zeros(cp2.shape)
#
#         co1[:rr, :] = cp1[:rr, :]
#         co2[:rr, :] = cp2[:rr, :]
#
#         co1[rr:rr + 1, :rc + 1] = cp1[rr:rr + 1, :rc + 1]
#         co2[rr:rr + 1, :rc + 1] = cp2[rr:rr + 1, :rc + 1]
#
#         co1[rr:rr + 1, rc + 1:] = cp2[rr:rr + 1, rc + 1:]
#         co2[rr:rr + 1, rc + 1:] = cp1[rr:rr + 1, rc + 1:]
#
#         co1[rr + 1:, :] = cp2[rr + 1:, :]
#         co2[rr + 1:, :] = cp1[rr + 1:, :]
#
#         return co1, co2
#
#     def vertical_crossover():
#         co1 = np.zeros(cp1.shape)
#         co2 = np.zeros(cp2.shape)
#
#         co1[:, :rc] = cp1[:, :rc]
#         co2[:, :rc] = cp2[:, :rc]
#
#         co1[:rr + 1, rc:rc + 1] = cp1[:rr + 1, rc:rc + 1]
#         co2[:rr + 1, rc:rc + 1] = cp2[:rr + 1, rr:rr + 1]
#
#         co1[rr + 1:, rc:rc + 1] = cp2[rr + 1:, rc:rc + 1]
#         co2[rr + 1:, rc:rc + 1] = cp1[rr + 1:, rc:rc + 1]
#
#         co1[:, rc + 1:] = cp2[:, rc + 1:]
#         co2[:, rc + 1:] = cp1[:, rc + 1:]
#
#         return co1, co2
#
#     if np.random.random() < Pc:
#         rr = np.random.randint(cp1.shape[0])
#         rc = np.random.randint(cp1.shape[1])
#
#         if np.random.random() > 0.5:
#             return horizontal_crossover()
#         else:
#             return vertical_crossover()
#     return ()
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     population = []
#     best_soln = None
#     best_fitness = float('-inf')
#     nq = NQueens(10)
#     # population.append(np.array([[1, 3, 0, 2]]))
#     for i in range(30):
#         population.append(nq.generate_random_config())
#
#     t = 0
#     while True:
#         if t == 30: break
#         print(best_soln, best_fitness)
#         lbc = best_fitness
#         for i in range(len(population)):
#             fitness = NQueens.evaluate_fitness(population[i])
#             if fitness > best_fitness:
#                 best_soln, best_fitness = population[i], fitness
#
#         if lbc == best_fitness:
#             t += 1
#         selected = stochastic_rank(population)
#
#         sums = sum(range(1, len(selected) + 1))
#         selected = [selected[np.random.choice(len(selected), p=[(i + 1) / sums for i in range(len(selected))])] for _ in range(len(population))]
#         children = []
#         for i in range(0, len(population), 2):
#             cp1, cp2 = selected[i], selected[i + 1]
#             for c in crossover(cp1, cp2):
#                 mutate(c)
#                 children.append(c)
#
#         population = children
#
#     # parent1 = np.random.randint(1, 10, (3, 4))
#     # parent2 = np.random.randint(1, 10, (3, 4))
#     # print("randomly generated parent1: \n", parent1)
#     # print("randomly generated parent2: \n", parent2)
#     # res = crossover(parent1, parent2)
#     # if res:
#     #     print("crossing over resulted in two children")
#     #     child1, child2 = res
#     #     print("child 1 before mutation: \n", child1)
#     #     mutate(child1)
#     #     print("child 1 after mutation: \n", child1)
#
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/
