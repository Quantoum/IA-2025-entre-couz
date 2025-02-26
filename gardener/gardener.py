from pycsp3 import *


def solve_gardener(instructions: list[list[int]]) -> list[list[int]]:
    # Put your code here
    
    return None

def verify_format(solution: list[list[int]], n: int):
    validity = True
    if (len(solution) != n):
        validity = False
        print("The number of rows in the solution is not equal to n")
    for i in range(len(solution)):
        if len(solution[i]) != n:
            validity = False
            print(f"Row {i} does not contain the right number of cells\n")
        for j in range(len(solution[i])):
            if (not isinstance(solution[i][j], int)):
                validity = False
                print(f"Cell in row {i} and column {j} is not an integer\n")

    return validity

def parse_instance(input_file: str) -> list[list[(int, int)]]:
    with open(input_file) as input:
        lines = input.readlines()
    n = int(lines[0].strip())
    instructions = []
    for line in lines[1:5]:
        instructions.append(list(map(int, line.strip().split(" "))))
        assert len(instructions[-1]) == n

    return instructions


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 gardener.py instance_path")
        sys.exit(1)

    instructions = parse_instance(sys.argv[1])

    solution = solve_gardener(instructions)
    if solution is not None:
        if verify_format(solution, len(instructions[0])):
            print("Solution format is valid")
        else:
            print("Solution format is invalid")
    else:
        print("No solution found")
    

