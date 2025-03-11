from pycsp3 import *


def solve_gardener(instructions: list[list[int]]) -> list[list[int]]:
    n = len(instructions[0])
    lines = VarArray(size=len(instructions[0]), dom=range(1, n + 1))
    columns = VarArray(size=len(instructions[1]), dom=range(1, n + 1))
    
    for i in range(n):
        satisfy(AllDifferent(lines[i]))
        satisfy(AllDifferent(columns[i]))
        
    for k in range(n):     
            #if n == 0:
            #    return []
            heights = VarArray(size=n, dom=range(1, n + 1))
            visible_left = VarArray(size=n, dom={0, 1})
            visible_right = VarArray(size=n, dom={0, 1})
            # Ensure all heights are unique
            satisfy(AllDifferent(heights))
            
            # The first hedge is always visible
            satisfy(visible_left[0] == 1)
            satisfy(visible_right[0] == 1)
            
            for i in range(1, n):
                conditions_left = [heights[i] > heights[j] for j in range(i)]
                conditions_right = [heights[i] < heights[j] for j in range(i)] 

                # Combine conditions with logical AND
                cond = conditions_left[0]
                for c in conditions_left[1:]: # verify that cond is greater than every other
                    cond = cond & c # if c is false (other one greater than him), then cond is false (and it never changes)
                satisfy(visible_left[i] == cond) # if the tower is visible
                
                cond = conditions_left[0]
                for c in conditions_left[1:]: # verify that cond is greater than every other
                    cond = cond & c # if c is false (other one greater than him), then cond is false (and it never changes)
                satisfy(visible_right[i] == cond) # if the tower is visible

            # Total visible must match the instruction
            satisfy(Sum(visible_left) == instructions[1][k])
            satisfy(Sum(visible_right) == instructions[2][k])
            

    if solve(solver=CHOCO) is SAT:
        return [heights[i].value for i in range(n)]
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
    

