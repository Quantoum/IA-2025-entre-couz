from pycsp3 import *


def solve_restricted_gardener(instruction: int, n: int) -> list[int]:
    if n == 0:
        return []
    heights = VarArray(size=n, dom=range(1, n + 1))
    visible = VarArray(size=n, dom={0, 1})

    # Ensure all heights are unique
    satisfy(AllDifferent(heights))
    
    # The first hedge is always visible
    satisfy(visible[0] == 1)

    for i in range(1, n):
        conditions = [heights[i] > heights[j] for j in range(i)]

        # Combine conditions with logical AND
        cond = conditions[0]
        for c in conditions[1:]: # verify that cond is greater than every other
            cond = cond & c # if c is false (other one greater than him), then cond is false (and it never changes)
        satisfy(visible[i] == cond) # if the tower is visible

    # Total visible must match the instruction
    satisfy(Sum(visible) == instruction)

    if solve(solver=CHOCO) is SAT:
        return [heights[i].value for i in range(n)]
    else:
        return None

def verify_format(solution: list[int], n: int):
    if len(solution) != n:
        print(f"The solution does not contain the right number of cells\n")
        for i in range(len(solution)):
            if (not isinstance(solution[i], int)):
                print(f"Cell at index {i} is not an integer\n")

def parse_instance(input_file: str) -> tuple[int, int]:
    with open(input_file, "r") as file:
        lines = file.readlines()
    n = int(lines[0].strip())
    instruction = int(lines[1].strip())

    return instruction, n


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 restricted_gardener.py instance_path")
        sys.exit(1)

    instruction, n  = parse_instance(sys.argv[1])


    solution = solve_restricted_gardener(instruction, n)
    if solution is not None:
        if (verify_format(solution, n)):
            print("Solution format is valid")
        else:
            print("Solution is invalid")
    else:
        print("No solution found")
