from pycsp3 import *

def solve_gardener(instructions):
    if not instructions or len(instructions) != 4 or not instructions[0]:
        return None
    n = len(instructions[0])
    
    # Create n x n grid
    grid = VarArray(size=(n, n), dom=range(1, n+1))
    
    # AllDifferent for rows and columns
    for i in range(n):
        satisfy(AllDifferent(grid[i]))
    for j in range(n):
        satisfy(AllDifferent([grid[i][j] for i in range(n)]))
    
    # Process rows (left/right visibility)
    for i in range(n):
        # Left visibility (unique ID per row)
        left_visible = VarArray(size=n, dom={0,1}, id=f"left_vis_{i}")
        satisfy(left_visible[0] == 1)
        for j in range(1, n):
            cond = conjunction(grid[i][j] > grid[i][k] for k in range(j))
            satisfy(left_visible[j] == cond)
        satisfy(Sum(left_visible) == instructions[1][i])
        
        # Right visibility (unique ID per row)
        right_visible = VarArray(size=n, dom={0,1}, id=f"right_vis_{i}")
        satisfy(right_visible[-1] == 1)
        for j in range(n-1):
            cond = conjunction(grid[i][j] > grid[i][k] for k in range(j+1, n))
            satisfy(right_visible[j] == cond)
        satisfy(Sum(right_visible) == instructions[2][i])
    
    # Process columns (top/bottom visibility)
    for j in range(n):
        # Top visibility (unique ID per column)
        top_visible = VarArray(size=n, dom={0,1}, id=f"top_vis_{j}")
        satisfy(top_visible[0] == 1)
        for i in range(1, n):
            cond = conjunction(grid[i][j] > grid[k][j] for k in range(i))
            satisfy(top_visible[i] == cond)
        satisfy(Sum(top_visible) == instructions[0][j])
        
        # Bottom visibility (unique ID per column)
        bottom_visible = VarArray(size=n, dom={0,1}, id=f"bot_vis_{j}")
        satisfy(bottom_visible[-1] == 1)
        for i in range(n-1):
            cond = conjunction(grid[i][j] > grid[k][j] for k in range(i+1, n))
            satisfy(bottom_visible[i] == cond)
        satisfy(Sum(bottom_visible) == instructions[3][j])
    
    if solve(solver=CHOCO) is SAT:
        return [[var.value for var in row] for row in grid]
    else:
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
    