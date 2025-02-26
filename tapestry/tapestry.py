from pycsp3 import *


def solve_tapestry(clues: list[list[(int, int)]]) -> list[list[(int, int)]]:
    # Put your code here

    return None

def verify_format(solution: list[list[(int, int)]], n: int):
    validity = True
    if (len(solution) != n):
        validity = False
        print("The number of rows in the solution is not equal to n")
    for i in range(len(solution)):
        if len(solution[i]) != n:
            validity = False
            print(f"Row {i} does not contain the right number of cells\n")
        for j in range(len(solution[i])):
            if (not isinstance(solution[i][j], tuple)):
                validity = False
                print(f"Cell in row {i} and column {j} is not a tuple\n")
            elif len(solution[i][j]) != 2:
                validity = False
                print(f"Cell in row {i} and column {j} does not contain the right number of values\n")
    return validity

def parse_instance(input_file: str) -> list[list[(int, int)]]:
    with open(input_file) as input:
        lines = input.readlines()
    n = int(lines[0].strip())
    clues = [[(0, 0) for i in range(n)] for j in range(n)]
    for line in lines[1:]:
        i, j, s, c = line.strip().split(" ")
        clues[int(i)][int(j)] = (int(s), int(c))
    return n, clues

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 tapestry.py instance_path")
        sys.exit(1)

    n, clues = parse_instance(sys.argv[1])
    
    solution = solve_tapestry(clues)
    if solution is not None:
        if (verify_format(solution, n)):
            print("Solution format is valid")
        else:
            print("Solution format is invalid")
    else:
        print("No solution found")

