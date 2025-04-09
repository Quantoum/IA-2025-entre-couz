# Exploration large number of childs in MCTS
EXPLO_CHILD_CONST = 100000

# Maximum number of iterations for MCTS to avoid deep recursion
MAX_MCTS_ITERATIONS = 1000
MAX_NORMALIZE_VALUE = 1000.0   # maximum 
MAX_POSSIBLE_MOVE_RANDOM = 30  # if the number of possible move exceeds this, ai plays random
MAX_SIZE_TRANSPOSITION = 100000  # Increased 10x from 10000 to 100000
LATE_GAME_LIMIT = 30           # number of turns after it's late game
MAX_DEPTH = 50                 # prevent excessive depth

# game phase thresholds
EARLY_GAME_THRESHOLD = 15
MID_GAME_THRESHOLD = 30
LATE_GAME_THRESHOLD = 45

# complexity tresholds
MID_COMPLEXITY_THRESHOLD = 15  # <= 15 moves is medium complexity
LOW_COMPLEXITY_THRESHOLD = 8   # <= 8 moves is low complexity
# > 15 moves is high complexity

# time thresholds
CRITICAL_TIME_THRESHOLD = 1    # <= 1s is critical
LOW_TIME_THRESHOLD = 5         # <= 5s is low
MEDIUM_TIME_THRESHOLD = 20     # <= 20s is medium

PREDETERMINED_START = True     # if the agent start with predetermined moves or not

# Alpha-Beta related constants
MAX_DEPTH_A_B_LOW_TIME = 2     # Reduced from 3 to 2 for quicker decisions under time pressure
MAX_DEPTH_A_B_NORMAL = 4       # Standard depth for alpha-beta

EXPLORATION_WEIGHT_END_GAME = 0.8 