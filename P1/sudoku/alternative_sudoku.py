from pycsp3 import *
from math import floor

clues = [[16, 0, 1, 2],
         [9, 3, 12],
         [11, 4, 5],
         [20, 6, 15, 24],
         [5, 7, 8],
         [9, 9, 18],
         [5, 10, 19],
         [15, 11, 20],
         [12, 13, 14],
         [8, 16, 17],
         [30, 21, 29, 30, 38, 39],
         [5, 22, 31],
         [8,23, 32],
         [12, 25, 26],
         [13, 27, 28],
         [3, 33, 34],
         [13, 35, 44],
         [12, 36, 45],
         [8, 37, 46, 47],
         [8, 40, 41],
         [9, 42, 43],
         [9, 48, 57],
         [9, 49, 58],
         [15, 50, 59],
         [20, 51, 52, 53],
         [13, 54, 63, 72], 
         [20, 55, 56, 65, 74],
         [6, 60, 61],
         [19, 62, 70, 71],
         [12, 64, 73],
         [9, 66, 75],
         [14, 67, 76],
         [8, 68, 77],
         [7, 69, 78],
         [13, 79, 80]]

# x[i][j] is the value at row i and col j
x = VarArray(size=[9, 9], dom=range(1, 10))

satisfy(
   # constraint 1
   [AllDifferent(x[i]) for i in range(9)],

   # constraint 2
   [AllDifferent(x[:, j]) for j in range(9)],

   # constraint 3
   [AllDifferent(x[i:i + 3, j:j + 3]) for i in [0, 3, 6] for j in [0, 3, 6]],

   # constraint 4
   [Sum([x[floor(i/9)][i%9] for i in clue[1:]]) == clue[0] for clue in clues]
)

# Solve the problem and print the solution if found
if solve(solver=CHOCO) is SAT:
    print("SATISFIABLE")
    print(values(x))
else:
    print("UNSATISFIABLE")