# import numpy as np
#
#
# class NQueens:
#     def __init__(self, n: int):
#         self.N = n
#
#     def generate_random_config(self):
#         return np.random.randint(0, self.N, (self.N, )).tolist()
#
#     @staticmethod
#     def evaluate_fitness(c):
#         count = 0
#         for r1 in range(len(c)):
#             for r2 in range(len(c)):
#                 if r1 == r2:
#                     continue
#                 if c[r1] != c[r2] and abs(r1 - r2) != abs(c[r1] - c[r2]):
#                     count += 1
#         return count
#         # return # -((c[0, 0] - 0) ** 2 + (c[0, c.shape[1] - 1] - c.shape[1] + 1) ** 2)
#
#     @staticmethod
#     def evaluate_penalty(c):
#         count = 0
#         for r1 in range(len(c)):
#             for r2 in range(len(c)):
#                 if r1 == r2:
#                     continue
#                 if c[r1] == c[r2] or abs(r1 - r2) == abs(c[r1] - c[r2]):
#                     count += 1
#         return count
#
#
# if __name__ == '__main__':
#     nq = NQueens(5)
#     ex = nq.generate_random_config()
#     print(ex)
#     print(NQueens.evaluate_fitness(ex))
#     print(NQueens.evaluate_penalty(ex))