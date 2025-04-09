#!/usr/bin/env python3
"""
Hybrid Agent for Fenix Game
Integrates Monte Carlo Tree Search with Alpha-Beta pruning
"""

import time
import random
import math
import numpy as np
from math import sqrt
from fenix import FenixAction
from agent import Agent

# Game phases treshold
EARLY_GAME_THRESHOLD = 15
MID_GAME_THRESHOLD = 30
LATE_GAME_THRESHOLD = 45

# Complexity thresholds
MID_COMPLEXITY_THRESHOLD = 15  # medium complexity
LOW_COMPLEXITY_THRESHOLD = 8   # low complexity

# Time thresholds
CRITICAL_TIME_THRESHOLD = 3    # critical
LOW_TIME_THRESHOLD = 10        # low
MEDIUM_TIME_THRESHOLD = 20     # medium

# MCTS constants
MAX_MCTS_ITERATIONS = 5000
MAX_DEPTH = 40
MAX_NORMALIZE_VALUE = 1000.0
EXPLORATION_WEIGHT_END_GAME = 0.8
MAX_POSSIBLE_MOVE_RANDOM = 30  # if too many moves, play random

# Alpha-Beta constants
MAX_DEPTH_A_B_LOW_TIME = 2
MAX_DEPTH_A_B_NORMAL = 3

# Transposition table settings
MAX_SIZE_TRANSPOSITION = 100000

# Use opening book?
PREDETERMINED_START = True

MAX_THINKING_TIME = 30  # Maximum allowed thinking time in seconds

# Debug mode
DEBUG_MODE = True

# Zobrist hashing for transposition table
_ZOBRIST_INITIALIZED = False
_zobrist_pieces = None
_zobrist_player = None

def _init_global_zobrist_tables():
    """Initialize global Zobrist tables with fixed seed."""
    global _ZOBRIST_INITIALIZED, _zobrist_pieces, _zobrist_player
    
    if not _ZOBRIST_INITIALIZED:
        # generating zobrists keys
        rng = np.random.default_rng(seed=42)
        # 2 players
        # 3 types of piece
        # 7 x 8 board
        _zobrist_pieces = rng.integers(0, 2**63, size=(2, 3, 7, 8), dtype=np.uint64)
        _zobrist_player = rng.integers(0, 2**63, size=2, dtype=np.uint64)
        _ZOBRIST_INITIALIZED = True

_init_global_zobrist_tables()

class TranspositionTable:
    """
    Transposition table (zobrist hash).
    Each entry contains:
    - hash_key: Zobrist hash of the position
    - value: best value found
    - depth: search depth
    - flag: EXACT, LOWER_BOUND, or UPPER_BOUND
    - best_move: best move found
    """

    EXACT = 0
    LOWER_BOUND = 1
    UPPER_BOUND = 2

    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.table = {}
        self.hits = 0
        self.misses = 0

    def compute_hash(self, state):
        """Compute Zobrist hash for a game state."""
        hash_value = 0
        
        # XOR the zobrist's keys
        current_player = state.to_move()
        if current_player == 1:
            hash_value ^= _zobrist_player[0]
        else:
            hash_value ^= _zobrist_player[1]
        
        # XOR in all pieces on the board
        for position, weight in state.pieces.items():
            player_index = 0 if weight > 0 else 1
            piece_type = min(2, abs(weight) - 1)
            x, y = position
            hash_value ^= _zobrist_pieces[player_index][piece_type][x][y]
        
        return int(hash_value)
        
    def get(self, state, depth, alpha, beta):
        """Look up a position in the trans table."""
        # Get hash
        hash_key = self.compute_hash(state)
        
        # Ensure hash key is a Python int
        if hasattr(hash_key, 'item'):
            hash_key = int(hash_key.item())
        else:
            hash_key = int(hash_key)
        
        # If the hash is in the table 
        if hash_key in self.table:
            entry = self.table[hash_key]
            self.hits += 1
            
            # Check if the entry is useful
            if entry['depth'] >= depth:
                if entry['flag'] == self.EXACT:
                    return entry['value'], entry['best_move']
                elif entry['flag'] == self.LOWER_BOUND and entry['value'] >= beta:
                    return entry['value'], entry['best_move']
                elif entry['flag'] == self.UPPER_BOUND and entry['value'] <= alpha:
                    return entry['value'], entry['best_move']
        else:
            self.misses += 1
            
        return None, None
        
    def put(self, state, depth, value, flag, best_move):
        """Store a position in the transposition table."""
        # Get hash
        hash_key = self.compute_hash(state)
        
        # Ensure hash key is a Python int
        if hasattr(hash_key, 'item'):
            hash_key = int(hash_key.item())
        else:
            hash_key = int(hash_key)
        
        # Check if we should replace an existing entry
        replace = True
        if hash_key in self.table:
            old_entry = self.table[hash_key]
            
            # Don't replace deeper searches with less good ones
            if old_entry['depth'] > depth:
                if old_entry['flag'] == self.EXACT and flag != self.EXACT:
                    replace = False
                else:
                    replace = False
        
        if replace:
            # Store entry
            self.table[hash_key] = {
                'depth': depth,
                'value': value,
                'flag': flag,
                'best_move': best_move
            }
        
        # If table is full, remove least valuable entries
        if len(self.table) >= self.max_size:
            keys = list(self.table.keys())
            num_to_remove = max(1, int(0.1 * self.max_size))
            for _ in range(num_to_remove):
                key_to_remove = np.random.choice(keys)
                if hasattr(key_to_remove, 'item'):
                    key_to_remove = int(key_to_remove.item())
                self.table.pop(key_to_remove, None)
        
    def get_hit_rate(self):
        """Calculate the hit rate of the transposition table."""
        total = self.hits + self.misses
        if total > 0:
            return self.hits / total
        return 0
        
    def clear(self):
        """Clear the transposition table."""
        self.table.clear()
        self.hits = 0
        self.misses = 0
    
    def get_entry(self, state):
        """Get an entry from the table directly using state."""
        hash_key = self.compute_hash(state)
        if hasattr(hash_key, 'item'):
            hash_key = int(hash_key.item())
        else:
            hash_key = int(hash_key)
            
        if hash_key in self.table:
            self.hits += 1
            return self.table[hash_key]
        else:
            self.misses += 1
            return None

    def store(self, state, wins, visits, value=None, best_action=None):
        """Store MCTS statistics in the table (used by MCTS nodes)."""
        hash_key = self.compute_hash(state)
        if hasattr(hash_key, 'item'):
            hash_key = int(hash_key.item())
        else:
            hash_key = int(hash_key)
            
        self.table[hash_key] = {
            'wins': wins,
            'visits': visits,
            'value': value,
            'best_move': best_action,
            'flag': self.EXACT,
            'depth': 0
        }
        
        # If table is full, remove least valuable entries
        if len(self.table) >= self.max_size:
            keys = list(self.table.keys())
            num_to_remove = max(1, int(0.1 * self.max_size))
            for _ in range(num_to_remove):
                key_to_remove = np.random.choice(keys)
                if hasattr(key_to_remove, 'item'):
                    key_to_remove = int(key_to_remove.item())
                self.table.pop(key_to_remove, None)
    
    def lookup(self, state):
        """Look up a state for MCTS (simplified interface)."""
        return self.get_entry(state)


class AlphaBeta:
    def __init__(self, player, max_depth=float('inf')):
        self.player = player
        self.max_depth = max_depth
        self.transposition_table = None
        self.nodes_evaluated = 0
        self.max_depth_reached = 0

    def alpha_beta_search(self, state):
        if self.transposition_table is None:
            raise ValueError("Transposition table not set for AlphaBeta agent.")
        
        self.nodes_evaluated = 0
        self.max_depth_reached = 0
        best_action = None
        
        # Track start time for absolute time limit
        self.start_time = time.time()
        if state.to_move() == self.player:
            best_value = -float('inf')
            for action in state.actions():
                value, _ = self.min_value(state.result(action), -float('inf'), float('inf'), 0)
                if value > best_value:
                    best_value = value
                    best_action = action
                    
                # Check if we've exceeded the time limit
                if time.time() - self.start_time > MAX_THINKING_TIME:
                    if DEBUG_MODE:
                        print(f"Alpha-Beta stopping: maximum time ({MAX_THINKING_TIME}s) reached after exploring {self.nodes_evaluated} nodes")
                    break
        else:
            best_value = float('inf')
            for action in state.actions():
                value, _ = self.max_value(state.result(action), -float('inf'), float('inf'), 0)
                if value < best_value:
                    best_value = value
                    best_action = action
                    
                # Check if we've exceeded the time limit
                if time.time() - self.start_time > MAX_THINKING_TIME:
                    if DEBUG_MODE:
                        print(f"Alpha-Beta stopping: maximum time ({MAX_THINKING_TIME}s) reached after exploring {self.nodes_evaluated} nodes")
                    break
        
        elapsed_time = time.time() - self.start_time
        if DEBUG_MODE and elapsed_time > MAX_THINKING_TIME * 0.9:  # if we used 90% or more of our time
            print(f"Alpha-Beta search used {elapsed_time:.2f}s out of {MAX_THINKING_TIME}s limit (max depth: {self.max_depth_reached})")
                    
        return best_action

    def max_value(self, state, alpha, beta, depth):
        self.nodes_evaluated += 1
        self.max_depth_reached = max(self.max_depth_reached, depth)
        
        # Check for time limit
        if hasattr(self, 'start_time') and time.time() - self.start_time > MAX_THINKING_TIME:
            return self.heuristics(state), None
        
        if state.is_terminal() or depth >= self.max_depth:
            return self.heuristics(state), None
        
        original_alpha = alpha 
        tt_value, tt_move = self.transposition_table.get(state, depth, alpha, beta)
        if tt_value is not None:
            return tt_value, tt_move
        
        value = -float('inf')
        best_action_for_node = None
        
        actions = state.actions()
        if tt_move is not None and tt_move in actions:
            # Prioritize move from transposition table
            actions.remove(tt_move)
            actions.insert(0, tt_move)
             
        for action in actions:
            v, _ = self.min_value(state.result(action), alpha, beta, depth + 1)
            if v > value:
                value = v
                best_action_for_node = action
                alpha = max(alpha, value)
                
            if value >= beta:
                self.transposition_table.put(state, depth, value, TranspositionTable.LOWER_BOUND, best_action_for_node)
                return value, best_action_for_node
                
            # Check for time limit
            if hasattr(self, 'start_time') and time.time() - self.start_time > MAX_THINKING_TIME:
                break
        
        if value <= original_alpha:
            flag = TranspositionTable.UPPER_BOUND
        else:
            flag = TranspositionTable.EXACT
        self.transposition_table.put(state, depth, value, flag, best_action_for_node)
        
        return value, best_action_for_node

    def min_value(self, state, alpha, beta, depth):
        self.nodes_evaluated += 1
        self.max_depth_reached = max(self.max_depth_reached, depth)
        
        # Check for time limit
        if hasattr(self, 'start_time') and time.time() - self.start_time > MAX_THINKING_TIME:
            return self.heuristics(state), None
        
        if state.is_terminal() or depth >= self.max_depth:
            return self.heuristics(state), None
            
        original_beta = beta
        tt_value, tt_move = self.transposition_table.get(state, depth, alpha, beta)
        if tt_value is not None:
            return tt_value, tt_move
        
        value = float('inf')
        best_action_for_node = None
        
        actions = state.actions()
        if tt_move is not None and tt_move in actions:
            # Prioritize move from transposition table
            actions.remove(tt_move)
            actions.insert(0, tt_move)
             
        for action in actions:
            v, _ = self.max_value(state.result(action), alpha, beta, depth + 1)
            if v < value:
                value = v
                best_action_for_node = action  
                beta = min(beta, value)  
                
            if value <= alpha:
                self.transposition_table.put(state, depth, value, TranspositionTable.UPPER_BOUND, best_action_for_node)
                return value, best_action_for_node
                
            # Check for time limit
            if hasattr(self, 'start_time') and time.time() - self.start_time > MAX_THINKING_TIME:
                break
            
        if value >= original_beta:
            flag = TranspositionTable.LOWER_BOUND
        else: 
            flag = TranspositionTable.EXACT
        self.transposition_table.put(state, depth, value, flag, best_action_for_node)
        
        return value, best_action_for_node

    def best_action(self, state):
        """Entry point for finding the best action."""
        return self.alpha_beta_search(state)
    
    def heuristics(self, state):
        if state.is_terminal():
            utility = state.utility(self.player)
            if utility == 1: return 10000 
            if utility == -1: return -10000
            return 0 
        
        score = 0
        score += 3 * self.materialHeuristic(state) 
        score += 1 * self.positionalHeuristic(state)
        return score
    
    def materialHeuristic(self, state):
        """Calculate material balance based on piece weights."""
        score = 0
        for position, weight in state.pieces.items():
            score += weight
        
        if self.player == -1:
            score = -score
        
        return score
    
    def positionalHeuristic(self, state):
        """Simple heuristic: counts pieces on the border."""
        border_positions = {(0,y) for y in range(8)} | {(6,y) for y in range(8)} | \
                        {(x,0) for x in range(7)} | {(x,7) for x in range(7)}
        
        on_border = 0
        opponent_on_border = 0
        
        for position, weight in state.pieces.items():
            if position in border_positions:
                is_our_piece = (weight > 0 and self.player == 1) or (weight < 0 and self.player == -1)
                
                if is_our_piece:
                    on_border += 1
                else:
                    opponent_on_border += 1
                    
        return on_border - opponent_on_border 


class BetterMCTSNode:
    def __init__(self, state, player, alpha_beta, trans_table=None,
                 parent=None, parent_action=None, max_iterations=MAX_MCTS_ITERATIONS,
                 exploration_weight=sqrt(2), rollout_depth=3, time_limit=None):
        self.state = state
        self.player = player
        self.parent = parent
        self.parent_action = parent_action
        self.max_iterations = max_iterations
        self.children = []
        self._number_of_visits = 0
        self._results = {1: 0, 0: 0, -1: 0}
        self._untried_actions = None
        self._untried_actions = self.untried_actions()
        self.alpha_beta = alpha_beta
        self.exploration_weight = exploration_weight
        self.rollout_depth = rollout_depth
        self.time_limit = time_limit
        self.start_time = time.time() if time_limit else None
        self.trans_table = trans_table
        
        # Check if this state is in the transposition table
        if self.trans_table is not None:
            entry = self.trans_table.lookup(state)
            if entry is not None:
                # Initialize with stored values if available
                self._number_of_visits = entry.get('visits', 0)
                wins = entry.get('wins', 0)
                if wins > 0:
                    self._results[1] = wins

    def is_time_up(self):
        """Check if the allocated time is finished"""
        if self.time_limit is None or self.start_time is None:
            return False
        return (time.time() - self.start_time) >= self.time_limit
    
    def untried_actions(self):
        """Get the list of untried actions from current state."""
        self._untried_actions = self.state.actions()
        return self._untried_actions
    
    def q(self):
        """Compute the difference between wins and loses."""
        wins = self._results[1]
        loses = self._results[-1]
        return wins - loses
    
    def n(self):
        """Return the number of times this node has been visited."""
        return self._number_of_visits
    
    def expand(self):
        """Generate the next state based on an untried action."""
        if not self._untried_actions:
            return None
            
        action = self._untried_actions.pop()
        next_state = self.state.result(action)
        child_node = BetterMCTSNode(
            state=next_state, 
            player=self.player, 
            alpha_beta=self.alpha_beta,
            trans_table=self.trans_table,
            parent=self, 
            parent_action=action, 
            max_iterations=self.max_iterations,
            exploration_weight=self.exploration_weight, 
            rollout_depth=self.rollout_depth, 
            time_limit=self.time_limit
        )
        self.children.append(child_node)
        return child_node
    
    def is_terminal_node(self):
        """Check if current state is terminal."""
        return self.state.is_terminal()
    
    def rollout(self):
        """Use Alpha-Beta for short-term tactical evaluation."""
        # Terminal state check
        if self.state.is_terminal():
            return self.state.utility(self.player)
        
        # Use alpha-beta for evaluation
        self.alpha_beta.max_depth = self.rollout_depth
        ab_value = self.alpha_beta.heuristics(self.state)
        
        # Normalize value between -1 and 1
        result = max(min(ab_value / MAX_NORMALIZE_VALUE, 1.0), -1.0)
        return result
    
    def backpropagate(self, result):
        """Update statistics for all nodes up to the root."""
        self._number_of_visits += 1
        
        # Convert float results to integer categories for dictionary keys
        # This handles the case where rollout returns a float value
        if isinstance(result, float):
            # Map float values to integer keys: 1 (win), 0 (draw), -1 (loss)
            if result > 0.001:
                result_key = 1  # positive values count as wins
            elif result < -0.001:
                result_key = -1  # negative values count as losses 
            else:
                result_key = 0  # values close to zero count as draws
        else:
            result_key = result  # already an integer
            
        self._results[result_key] += 1
        
        # Update transposition table
        if self.trans_table is not None:
            self.trans_table.store(
                self.state,
                self._results[1],  # wins
                self._number_of_visits,  # visits
                value=self.q() / max(1, self._number_of_visits),  # average value
                best_action=self.get_best_action()  # best action
            )
        
        # Propagate up the tree
        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expanded(self):
        """Check if all possible actions have been tried."""
        return len(self._untried_actions) == 0
    
    def get_best_action(self):
        """Get the best action based on visit counts."""
        if not self.children:
            return None
            
        visits = {child.parent_action: child.n() for child in self.children}
        if not visits:
            return None
            
        return max(visits.items(), key=lambda x: x[1])[0]
    
    def best_child(self, c_param=None):
        """Select the best child node using UCB."""
        if c_param is None:
            c_param = self.exploration_weight
            
        # Check for terminal winning states
        for child in self.children:
            if child.is_terminal_node() and child.state.utility(self.player) == 1:
                return child
        
        # No children case
        if not self.children:
            return None

        # Calculate UCB for each child
        best_score = float('-inf')
        best_child = None
        
        for child in self.children:
            if child.n() == 0:
                # Unexplored nodes get high priority
                score = float('inf')
            else:
                # UCB formula
                exploitation = child.q() / child.n()
                exploration = c_param * math.sqrt(2 * math.log(self.n()) / child.n())
                
                # Reduce exploration in late game
                if self.state.turn >= LATE_GAME_THRESHOLD:
                    exploration *= 0.5
                
                score = exploitation + exploration
            
            # Update best if score is higher
            if best_child is None or score > best_score:
                best_score = score
                best_child = child
                
        return best_child
    
    def _tree_policy(self):
        """Select or expand nodes according to the tree policy."""
        current_node = self
        depth = 0
        
        while not current_node.is_terminal_node() and depth < MAX_DEPTH:
            depth += 1
            
            # Check time limit
            if self.is_time_up():
                return current_node
                
            # Expand if not fully expanded
            if not current_node.is_fully_expanded():
                expanded = current_node.expand()
                if expanded:
                    return expanded
                
            # Select best child
            next_node = current_node.best_child()
            if next_node is None:
                return current_node
            current_node = next_node
                
        return current_node
    
    def best_action(self):
        """Run MCTS and return the best action."""
        iterations_completed = 0
        absolute_start_time = time.time()  # Absolute start time for hard 50-second limit
        
        while iterations_completed < self.max_iterations:
            # Check time limit set in parameters
            if self.is_time_up():
                if DEBUG_MODE:
                    print(f"MCTS stopping: time limit reached after {iterations_completed} iterations")
                break
                
            # Check absolute time limit (50 seconds)
            current_time = time.time()
            if current_time - absolute_start_time > MAX_THINKING_TIME:
                if DEBUG_MODE:
                    print(f"MCTS stopping: absolute max time ({MAX_THINKING_TIME}s) reached after {iterations_completed} iterations")
                break
                
            # 1. Selection and expansion
            v = self._tree_policy()
            
            # 2. Simulation
            reward = v.rollout()
            
            # 3. Backpropagation
            v.backpropagate(reward)
            
            iterations_completed += 1
        
        # If no iterations completed, expand once
        if iterations_completed == 0 and not self.children:
            self.expand()
        
        # If still no children, return self
        if not self.children:
            if DEBUG_MODE:
                print("No children generated during MCTS search")
            return self
            
        # Select best child based on visit count
        visits = [child.n() for child in self.children]
        
        # Check if any visits occurred
        if sum(visits) == 0:
            # Random selection if no visits
            best_child = random.choice(self.children) if self.children else self
        else:
            # Select most visited child
            best_child = self.children[np.argmax(visits)]
            
        elapsed_time = time.time() - absolute_start_time
        if elapsed_time > MAX_THINKING_TIME * 0.9:  # if we used 90% or more of our time
            if DEBUG_MODE:
                print(f"MCTS search used {elapsed_time:.2f}s out of {MAX_THINKING_TIME}s limit ({iterations_completed} iterations)")
            return best_child
            
        return best_child


class monAgent(Agent):
    def __init__(self, player):
        super().__init__(player)
        
        # Transposition table
        self.trans_table = TranspositionTable(max_size=MAX_SIZE_TRANSPOSITION)
        
        # Performance tracking
        self.strategy_history = []
        self.time_usage_history = []
        self.last_execution_time = 0
        self.wins = 0
        self.losses = 0
        self.avg_ab_depth = []
        self.avg_mcts_iterations = []
        self.last_time_check = time.time()
        
        # Opening book
        self.opening_book = None
        if PREDETERMINED_START:
            self.opening_book = self._create_opening_book()
        
    def _create_opening_book(self):
        """Create an opening book with predetermined moves."""
        if self.player == 1:  # red player
            return {
                0: ((1,0),(0,0)),  # Move general towards the corner
                2: ((0,1),(0,0)),  # Move king on top of general in corner
                4: ((1,1),(2,1)),  # Move another general outward
                6: ((0,2),(0,3)),  # Move general to middle border
                8: ((1,2),(2,2)),  # Move another general outward
            }
        else:  # black player
            return {
                1: ((5,7),(6,7)),  # Move general towards the corner
                3: ((6,6),(6,7)),  # Move king on top of general in corner
                5: ((5,6),(4,6)),  # Move another general outward
                7: ((5,5),(4,5)),  # Move other general outward
                9: ((6,5),(6,4)),  # Move general to middle border
            }
            
    def determine_game_phase(self, state):
        """Determine the current game phase."""
        # Opening phase
        if PREDETERMINED_START and state.turn in self.opening_book:
            return 'opening'
            
        # Early game
        if state.turn < EARLY_GAME_THRESHOLD:
            return 'early'
            
        # Late game
        if state.turn >= LATE_GAME_THRESHOLD:
            return 'late'
            
        # Mid game
        return 'mid'
        
    def evaluate_complexity(self, state):
        """Evaluate position complexity based on legal moves and pieces."""
        legal_moves = state.actions()
        num_moves = len(legal_moves)
        
        # Complexity based on move count
        if num_moves <= LOW_COMPLEXITY_THRESHOLD:
            complexity = 'low'
        elif num_moves <= MID_COMPLEXITY_THRESHOLD:
            complexity = 'medium'
        else:
            complexity = 'high'
            
        # Adjust for piece count
        piece_count = len(state.pieces)
        if piece_count <= 5:
            complexity = 'low'  # Few pieces usually means simpler positions
            
        return complexity
        
    def evaluate_time_situation(self, remaining_time):
        """Evaluate the time situation."""
        if remaining_time <= CRITICAL_TIME_THRESHOLD:
            return 'critical'
        elif remaining_time <= LOW_TIME_THRESHOLD:
            return 'low'
        elif remaining_time <= MEDIUM_TIME_THRESHOLD:
            return 'medium'
        else:
            return 'high'
            
    def select_strategy(self, state, remaining_time):
        """Select the best strategy based on game state and time."""
        game_phase = self.determine_game_phase(state)
        complexity = self.evaluate_complexity(state)
        time_situation = self.evaluate_time_situation(remaining_time)
        
        # Opening book moves
        if game_phase == 'opening':
            return {
                'name': 'opening',
                'params': {}
            }
            
        # Critical time situation - use random
        if time_situation == 'critical':
            return {
                'name': 'random',
                'params': {}
            }
            
        # Low time - use fast alpha-beta
        if time_situation == 'low':
            return {
                'name': 'alpha_beta',
                'params': {
                    'max_depth': MAX_DEPTH_A_B_LOW_TIME,
                    'time_limit': remaining_time * 0.8
                }
            }
            
        # Strategy based on game phase and complexity
        if game_phase == 'early':
            if complexity == 'low':
                return {
                    'name': 'alpha_beta',
                    'params': {
                        'max_depth': 7,
                        'time_limit': remaining_time * 0.8
                    }
                }
            else:
                return {
                    'name': 'mcts',
                    'params': {
                        'max_iterations': MAX_MCTS_ITERATIONS,
                        'rollout_depth': 2,
                        'time_limit': remaining_time * 0.8
                    }
                }
        elif game_phase == 'mid':
            if complexity == 'high':
                return {
                    'name': 'mcts',
                    'params': {
                        'max_iterations': MAX_MCTS_ITERATIONS,
                        'rollout_depth': 3,
                        'time_limit': remaining_time * 0.8,
                        'exploration_weight': 1.4
                    }
                }
            else:
                return {
                    'name': 'alpha_beta',
                    'params': {
                        'max_depth': 6,
                        'time_limit': remaining_time * 0.8
                    }
                }
        else:  # late game
            if complexity == 'low':
                return {
                    'name': 'alpha_beta',
                    'params': {
                        'max_depth': 8,
                        'time_limit': remaining_time * 0.8
                    }
                }
            else:
                return {
                    'name': 'mcts',
                    'params': {
                        'max_iterations': MAX_MCTS_ITERATIONS,
                        'rollout_depth': 4,
                        'time_limit': remaining_time * 0.8,
                        'exploration_weight': EXPLORATION_WEIGHT_END_GAME
                    }
                }
                
    def execute_opening_move(self, state):
        """Execute a move from the opening book."""
        move_tuple = self.opening_book.get(state.turn)
        if not move_tuple:
            if DEBUG_MODE:
                print(f"No opening move found for turn {state.turn}")
            return self.execute_alpha_beta(state, {'max_depth': 4})
            
        start_pos, end_pos = move_tuple
        action = FenixAction(start_pos, end_pos, removed=frozenset())
        
        # Verify validity
        legal_moves = state.actions()
        if action in legal_moves:
            return action
            
        # Fall back to Alpha-Beta if opening move invalid
        if DEBUG_MODE:
            print(f"Opening book move {action} not valid")
        return self.execute_alpha_beta(state, {'max_depth': 4})
        
    def execute_random(self, state):
        """Execute a random move selection."""
        legal_moves = state.actions()
        if not legal_moves:
            raise Exception("No legal moves available!")
            
        return random.choice(legal_moves)
        
    def execute_alpha_beta(self, state, params):
        """Execute Alpha-Beta search with the given parameters."""
        max_depth = params.get('max_depth', 4)
        time_limit = params.get('time_limit', MAX_THINKING_TIME)
        
        alpha_beta = AlphaBeta(self.player, max_depth=max_depth)
        alpha_beta.transposition_table = self.trans_table
        
        start_time = time.time()
        
        try:
            action = alpha_beta.best_action(state)
            
            # Record statistics
            self.avg_ab_depth.append(alpha_beta.max_depth_reached)
            
            # Check time limit
            elapsed = time.time() - start_time
            if time_limit and elapsed > time_limit:
                if DEBUG_MODE:
                    print(f"Warning: Alpha-Beta exceeded time limit: {elapsed:.2f}s > {time_limit:.2f}s")
                # Fall back to faster search
                alpha_beta.max_depth = MAX_DEPTH_A_B_LOW_TIME
                action = alpha_beta.best_action(state)
            
            return action
            
        except Exception as e:
            if DEBUG_MODE:
                print(f"Error in Alpha-Beta search: {e}")
            return self.execute_random(state)
    
    def execute_mcts(self, state, params):
        """Execute MCTS search with the given parameters."""
        max_iterations = params.get('max_iterations', MAX_MCTS_ITERATIONS)
        rollout_depth = params.get('rollout_depth', 3)
        time_limit = params.get('time_limit', None)
        exploration_weight = params.get('exploration_weight', sqrt(2))
        
        self.last_time_check = time.time()
        
        # Set up Alpha-Beta for rollouts
        alpha_beta = AlphaBeta(self.player, max_depth=rollout_depth)
        alpha_beta.transposition_table = self.trans_table
        
        try:
            # Create MCTS root
            mcts_root = BetterMCTSNode(
                state=state,
                player=self.player,
                alpha_beta=alpha_beta,
                trans_table=self.trans_table,
                max_iterations=max_iterations,
                rollout_depth=rollout_depth,
                time_limit=time_limit,
                exploration_weight=exploration_weight
            )
            
            # Run MCTS
            selected_node = mcts_root.best_action()
            
            # Record statistics - estimate iterations by visit count
            iterations_estimate = mcts_root.n()
            self.avg_mcts_iterations.append(iterations_estimate)
            
            # Check if selected_node is valid and has a parent_action
            if selected_node is None or selected_node == mcts_root or selected_node.parent_action is None:
                if DEBUG_MODE:
                    print("MCTS returned root node or node without parent action, using random fallback")
                return self.execute_random(state)
                
            # Extract action
            action = selected_node.parent_action
            return action
            
        except Exception as e:
            if DEBUG_MODE:
                print(f"MCTS failed: {str(e)}")
            # Fall back to Alpha-Beta
            return self.execute_alpha_beta(state, {'max_depth': 3, 'time_limit': time_limit})
    
    def act(self, state, remaining_time):
        """Choose an action for the current state."""
        start_time = time.time()
        self.last_time_check = start_time
        
        # Get legal moves
        legal_moves = state.actions()
        if not legal_moves:
            if DEBUG_MODE:
                print("No legal moves available!")
            return None
            
        # If only one legal move, play it immediately
        if len(legal_moves) == 1:
            # if DEBUG_MODE:
            #     print(f"Only one legal move available - playing it immediately")
            return legal_moves[0]
        
        try:
            # Select strategy
            strategy_info = self.select_strategy(state, remaining_time)
            strategy_name = strategy_info['name']
            strategy_params = strategy_info['params']
            
            if DEBUG_MODE:
                print(f"Selected strategy: {strategy_name}")
            
            # Record strategy choice
            self.strategy_history.append({
                'turn': state.turn,
                'strategy': strategy_name,
                'time_remaining': remaining_time,
                'num_legal_moves': len(legal_moves),
                'num_pieces': len(state.pieces)
            })
            
            # Execute selected strategy
            action = None
            if strategy_name == 'opening':
                action = self.execute_opening_move(state)
            elif strategy_name == 'random':
                action = self.execute_random(state)
            elif strategy_name == 'alpha_beta':
                action = self.execute_alpha_beta(state, strategy_params)
            elif strategy_name == 'mcts':
                print("Running mcts")
                action = self.execute_mcts(state, strategy_params)
            else:
                action = self.execute_random(state)
                
            # Record execution time
            execution_time = time.time() - start_time
            self.last_execution_time = execution_time
            self.time_usage_history.append(execution_time)
            
            return action
            
        except Exception as e:
            if DEBUG_MODE:
                print(f"Error in act method: {e}")
            return random.choice(legal_moves)
    
    def get_performance_summary(self):
        """Return a summary of the agent's performance."""
        strategy_counts = {}
        for record in self.strategy_history:
            strategy = record['strategy']
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            
        avg_time = 0
        if self.time_usage_history:
            avg_time = sum(self.time_usage_history) / len(self.time_usage_history)
            
        avg_ab_depth_value = 0
        if self.avg_ab_depth:
            avg_ab_depth_value = sum(self.avg_ab_depth) / len(self.avg_ab_depth)
            
        avg_mcts_iter_value = 0
        if self.avg_mcts_iterations:
            avg_mcts_iter_value = sum(self.avg_mcts_iterations) / len(self.avg_mcts_iterations)
            
        tt_hit_rate = self.trans_table.get_hit_rate()
            
        return {
            'strategy_usage': strategy_counts,
            'avg_time_per_move': round(avg_time, 2),
            'avg_ab_depth': avg_ab_depth_value,
            'avg_mcts_iterations': avg_mcts_iter_value,
            'tt_hit_rate': round(tt_hit_rate, 3),
            'tt_size': len(self.trans_table.table)
        }