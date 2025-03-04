from pycsp3 import *

def solve_tapestry(clues):
    n = len(clues)
    
    # Shapes and colors. format inupt : (s, c)
    shapes = VarArray(size=[n, n], dom=range(1, n+1))
    colors = VarArray(size=[n, n], dom=range(1, n+1))
    
    # constraints
    for i in range(n):
        # rows with s and c
        satisfy(AllDifferent(shapes[i]))
        satisfy(AllDifferent(colors[i]))
        # cols with s and c
        satisfy(AllDifferent(shapes[:, i]))
        satisfy(AllDifferent(colors[:, i]))
    
    # fill with clues
    for i in range(n):
        for j in range(n):
            c_shape, c_color = clues[i][j]
            if c_shape == 0 or c_color == 0: # ignore the case (empty clue)
                continue
            # fill the values
            satisfy(shapes[i][j] == c_shape)
            satisfy(colors[i][j] == c_color)
    
    # encode bcz VarArray cannot f*ck tuples  (linear encoding)
    encoded_pairs = [shapes[i][j] * (n + 1) + colors[i][j] for i in range(n) for j in range(n)]
    satisfy(AllDifferent(encoded_pairs)) # each pair !=
    
    # solve model
    if solve(solver=CHOCO) is SAT:
        # reconstruct the tuples
        solution = [[(shapes[i][j].value, colors[i][j].value) for j in range(n)] for i in range(n)]
        return solution
    else:
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
    clues = [[(0, 0) for _ in range(n)] for _ in range(n)]
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

