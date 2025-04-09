#!/usr/bin/env python3
"""
Hybrid Agent for Fenix Game
Integrates Monte Carlo Tree Search with Alpha-Beta pruning
"""

import time
import math

import numpy as np

from fenix import FenixAction

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

DEBUG_MODE = False

_ZOBRIST_INITIALIZED = False
_zobrist_pieces = None
_zobrist_player = None

def _init_global_zobrist_tables():
    """Initialize global Zobrist tables with fixed seed."""
    global _ZOBRIST_INITIALIZED, _zobrist_pieces, _zobrist_player
    
    if not _ZOBRIST_INITIALIZED:
        rng = np.random.default_rng(seed=42)
        
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
        """
        Compute Zobrist hash for a game state.
        Returns the Zobrist hash value
        """
        hash_value = 0
        
        # XOR in the player to move
        current_player = state.to_move()
        if current_player == 1:
            hash_value ^= _zobrist_player[0]  # Use global table
        else:
            hash_value ^= _zobrist_player[1]  # Use global table
        
        # XOR in all pieces on the board
        for position, weight in state.pieces.items():
            player_index = 0 if weight > 0 else 1
            piece_type = min(2, abs(weight) - 1)
            x, y = position
            hash_value ^= _zobrist_pieces[player_index][piece_type][x][y]
        
        return int(hash_value)
        
    def get(self, state, depth, alpha, beta):
        """
        Look up a position in the trans table.
        Returns: (value, best_move) if found and useful, None otherwise
        """
        # get hash
        hash_key = self.compute_hash(state)
        
        if hasattr(hash_key, 'item'):
            hash_key = hash_key.item()
        
        # if the hash is in the table 
        if hash_key in self.table:
            entry = self.table[hash_key]
            
            # Always count a hit whenever the position is found, regardless of usefulness
            self.hits += 1
            
            # Check if the entry is useful:
            if entry['depth'] >= depth:
                if entry['flag'] == self.EXACT:
                    # Exact value is always useful
                    return entry['value'], entry['best_move']
                elif entry['flag'] == self.LOWER_BOUND and entry['value'] >= beta:
                    # Lower bound that causes a beta cutoff
                    return entry['value'], entry['best_move']
                elif entry['flag'] == self.UPPER_BOUND and entry['value'] <= alpha:
                    # Upper bound that causes an alpha cutoff
                    return entry['value'], entry['best_move']
        else:
            self.misses += 1 # add a miss
            
        return None, None # not found or useless
        
    def put(self, state, depth, value, flag, best_move):
        """
        Store a position in the transposition table.
        Uses replacement strategy to prefer deeper searches and exact values.
        """
        # Get the hash 
        hash_key = self.compute_hash(state)
        
        if hasattr(hash_key, 'item'):
            hash_key = hash_key.item()
        
        replace = True
        if hash_key in self.table:
            old_entry = self.table[hash_key]
            
            # Don't replace deeper searches with less good ones
            if old_entry['depth'] > depth:
                # Prefer exact values over bounds
                if old_entry['flag'] == self.EXACT and flag != self.EXACT:
                    replace = False
                # Otherwise only keep the deeper search
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
            # Simple strategy: Remove a random entry
            # -> track entry age and replace oldest ?
            keys = list(self.table.keys())
            # remove 10% when reaching end
            num_to_remove = max(1, int(0.1 * self.max_size))
            for _ in range(num_to_remove):
                self.table.pop(np.random.choice(keys))
        
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
    
    def get_stats(self):
        """Get detailed statistics about the transposition table."""
        entry_types = {
            self.EXACT: 0,
            self.LOWER_BOUND: 0, 
            self.UPPER_BOUND: 0
        }
        depths = {}
        
        # Count entry types and depths
        for entry in self.table.values():
            entry_types[entry['flag']] += 1
            depth = entry['depth']
            depths[depth] = depths.get(depth, 0) + 1
            
        return {
            'size': len(self.table),
            'max_size': self.max_size,
            'hit_rate': self.get_hit_rate(),
            'hits': self.hits,
            'misses': self.misses,
            'entry_types': entry_types,
            'depths': depths
        }
    
    def print_stats(self):
        """Print detailed statistics about the transposition table."""

        if not DEBUG_MODE:
            print("DEBUG mode deactivated")
            return
        stats = self.get_stats()
        
        print("\n==== Transposition Table Statistics ====")
        print(f"Size: {stats['size']}/{stats['max_size']} ({stats['size']/stats['max_size']*100:.1f}% full)")
        print(f"Hit rate: {stats['hit_rate']:.2%} ({stats['hits']} hits, {stats['misses']} misses)")
        
        # Print entry types
        print("\nEntry Types:")
        types_labels = {
            self.EXACT: "EXACT",
            self.LOWER_BOUND: "LOWER_BOUND", 
            self.UPPER_BOUND: "UPPER_BOUND"
        }
        for flag, count in stats['entry_types'].items():
            print(f"  {types_labels[flag]}: {count} ({count/max(1, stats['size'])*100:.1f}%)")
            
        # Print depth distribution (at most 5 entries to avoid clutter)
        print("\nDepth Distribution (top 5):")
        sorted_depths = sorted(stats['depths'].items(), key=lambda x: x[1], reverse=True)
        for depth, count in sorted_depths[:5]:
            print(f"  Depth {depth}: {count} entries ({count/max(1, stats['size'])*100:.1f}%)")
            
        print("=======================================\n")

    def get_entry(self, state):
        hash_key = self.compute_hash(state)
        
        if hasattr(hash_key, 'item'):
            hash_key = hash_key.item()
            
        return self.table.get(hash_key)
    

#
# LINFO 1361 - Artificial Intelligence 
# Fenix game - April 2025
# Author: Arnaud Ullens, Quentin de Pierpont
# 

class AlphaBeta:
    def __init__(self, player, max_depth=float('inf')):
        self.player = player
        self.max_depth = max_depth
        self.transposition_table = None 

    def alpha_beta_search(self, state):
        if self.transposition_table is None:
            raise ValueError("Transposition table not set for AlphaBeta agent.")
            
        best_action = None
        if state.to_move() == self.player:
            best_value = -float('inf')
            for action in state.actions():
                value, _ = self.min_value(state.result(action), -float('inf'), float('inf'), 0)
                if value > best_value:
                    best_value = value
                    best_action = action
        else:
            best_value = float('inf')
            for action in state.actions():
                value, _ = self.max_value(state.result(action), -float('inf'), float('inf'), 0)
                if value < best_value:
                    best_value = value
                    best_action = action
                    
        return best_action

    def max_value(self, state, alpha, beta, depth):
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
        
        if value <= original_alpha:
            flag = TranspositionTable.UPPER_BOUND
        else:
            flag = TranspositionTable.EXACT
        self.transposition_table.put(state, depth, value, flag, best_action_for_node)
        
        return value, best_action_for_node

    def min_value(self, state, alpha, beta, depth):
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
        """Calculates material balance based on piece weights."""
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
    

#
# LINFO 1361 - Artificial Intelligence 
# Fenix game - April 2025
# Author: Arnaud Ullens, Quentin de Pierpont
# 


class MonteCarloTreeSearchNode():
    def __init__(self, state, player, parent=None, parent_action=None, max_iterations=100):
        self.state = state # board state
        self.player = player # player that has to move
        self.parent = parent # None for root node, equals to the node it's derived from/
        self.parent_action = parent_action # None for root node, equals action which it's parent carried out.
        self.max_iterations = max_iterations
        self.children = [] # all possible actions from the current node
        self._number_of_visits = 0 # number of time a current node is visited (back-propagation)
        self._results = {}
        self._results[1] = 0
        self._results[0] = 0
        self._results[-1] = 0
        self._untried_actions = None
        self._untried_actions = self.untried_actions() # list of all possible actions
    
    def untried_actions(self):
        """
            This function gets the list of untried actions from a given state.
            @return : list of untried actions
        """
        self._untried_actions = self.state.actions()
        return self._untried_actions
    
    def q(self):
        """
            This function computes the difference between wins and loses.
            @return : wins - loses
        """
        wins = self._results[1]
        loses = self._results[-1]
        return wins - loses
    
    def n(self):
        """
            This function computes the number of times each node is visited.
            @return: the number of visits
        """
        return self._number_of_visits
    
    def expand(self):
        """
            This function generates the next state (from the present state) depending
            on the action that has been carried out. 
            @return: The child node, a.k.a the next state.
        """
        action = self._untried_actions.pop()
        next_state = self.state.result(action)
        child_node = MonteCarloTreeSearchNode(next_state, player=self.player, parent=self, parent_action=action)
        self.children.append(child_node)
        return child_node
    
    def is_terminal_node(self):
        """
            This function checks wheter the state is final or not.
            @return: [bool] current not is terminal.
        """
        return self.state.is_terminal()
    
    def rollout(self):
        """
            This function generates an entire game from the current state (until there is an outcome).
            @return: the outcome of the game. (win=1, loss=-1, tie=0) -> depending on policy."""
        current_rollout_state = self.state

        while not current_rollout_state.is_terminal():
            possible_moves = current_rollout_state.actions()
            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.result(action)
        result = current_rollout_state.utility(self.player)
        return result
    
    def backpropagate(self, result):
        """
            This function computes all the statistics for the nodes that are updated. Until the parent
            node is reached, the number of visits is incremented. It increment by 1 the number of wins
            (or losses) depending on the result. Recursive.
        """
        self._number_of_visits += 1
        self._results[result] += 1
        if self.parent:
            self.parent.backpropagate(result)
    
    def is_fully_expanded(self):
        """
            This function is checking whether the node is fully expanded or not. All the actions are
            popped out of _untried_actions one by one. When it's empty, it's fully expanded.
            @return: whether the node is fully expanded or not.
        """
        return len(self._untried_actions) == 0
    
    def best_child(self, c_param=0.1):
        """
            This function selects the best child out of the children (once fully expanded).
            See the formula explanation in the associated documentation.
            @return: The best child.
        """
        choices_weights = []
        for c in self.children:
            if c.n() == 0:
                # unexplored nodes; ensure they are explored by adding large value
                choices_weights.append(EXPLO_CHILD_CONST)
            else:
                # UCT for explored nodes
                exploitation = c.q() / c.n()
                exploration = c_param * np.sqrt(2 * np.log(self.n()) / c.n())
                weight = exploitation + exploration
                choices_weights.append(weight)
        return self.children[np.argmax(choices_weights)]
    
    def rollout_policy(self, possible_moves):
        """
            This function selects a random move in all possible moves. It's an "example payout".
            Can be improved by heuristic, and/or Deep RL !
            @return: the selected move.
        """
        return np.random.choice(possible_moves)

    def _tree_policy(self):
        """
            This function selects the node to run rollout. Returns the best child, or expand the
            node.
            @return: the node, expanded, or the best child.
        """
        current_node = self
        depth = 0
        max_depth = 50
        while not current_node.is_terminal_node() and depth < max_depth:
            depth += 1
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node
    
    def best_action(self):
        """
            This function compute the best possible move.
            @return: the best action
        """
        for i in range(self.max_iterations):
            # 1 : Selection and expansion (find a node to simulate from)
            v = self._tree_policy()
            # 2 : Simulation (play out to terminal state)
            reward = v.rollout()
            # 3 : Backpropagation (update statistics up the tree)
            v.backpropagate(reward)
        best = self.best_child(c_param=np.sqrt(2))
        return best
    



class BetterMCTSNode(MonteCarloTreeSearchNode):
    def __init__(self, state, player, alpha_beta, 
                 parent=None, parent_action=None, max_iterations=MAX_MCTS_ITERATIONS,
                 exploration_weight=np.sqrt(2), rollout_depth=3, time_limit=None):
        super().__init__(state, player, parent, parent_action, max_iterations)
        self.alpha_beta = alpha_beta
        self.exploration_weight = exploration_weight
        self.rollout_depth = rollout_depth
        self.time_limit = time_limit
        self.start_time = time.time() if time_limit else None

    def is_time_up(self):
        """Check if the allocated time is finished"""
        return (time.time() - self.start_time) >= self.time_limit
    
    def rollout(self):
        """
            Alpha-Beta for short-term tactical evaluation
        """
        # if terminal state -> return utility
        if self.state.is_terminal():
            result = self.state.utility(self.player)
            return result
            
        # non-terminal -> alpha-beta

        # set depth
        self.alpha_beta.max_depth = self.rollout_depth
        
        # alpha-beta
        ab_value = self.alpha_beta.heuristics(self.state)
        
        # normalize value to set between -1 and 1
        result = self._normalize_value(ab_value)
        return result
    
    def _normalize_value(self, value):
        """
        Convert an Alpha-Beta evaluation to a result between -1 and 1
        """
        # assume values are between -1000 and 1000
        
        return max(min(value / MAX_NORMALIZE_VALUE, 1.0), -1.0)
    
    def rollout_policy(self, possible_moves):
        """
        Better rollout policy that uses heuristics to guide the selection
        """
        # erro handling
        if not possible_moves:
            return None
        
        if len(possible_moves) == 1: # one move possible, no choice
            print("Only one move possible, playing it !")
            return possible_moves[0]
            
        # no time, get random to survive !
        if self.is_time_up() or len(possible_moves) > MAX_POSSIBLE_MOVE_RANDOM:
            return np.random.choice(possible_moves)
        
        # evaluate with heuristic
        move_values = []
        adjusted_values = []
        
        for move in possible_moves:
            # sometimes it bugs, so try-except resolves this. But the moves_values.append(0) is maybe not good, even if it's neutral
            try:
                next_state = self.state.result(move)
                m_value = self.alpha_beta.materialHeuristic(next_state)
                p_value = self.alpha_beta.positionalHeuristic(next_state)
                value = 3 * m_value + p_value
                move_values.append(value)
            except Exception as e:
                print(f"Warning: Error evaluating move {move}: {e}")
                move_values.append(0)  # neutral
        
        # if all eval failed -> use random selection
        if not move_values or all(v == 0 for v in move_values):
            return np.random.choice(possible_moves)
            
        # weighted randomness -> avoid local optima, exploration, compensate innacuracies in heuristic
        min_val = min(move_values)
        max_val = max(move_values)
        
        # all same values -> choose randomly -> optimized to avoid this ?
        if max_val == min_val:
            return np.random.choice(possible_moves)
            
        # weighted random selection
        # normalize into positive range -> probabilistic selection
        # no negative weights (it would broke the probability)
        # reduce dominance from extreme
        for value in move_values:
            if self.player == 1:
                adjusted_values.append(value - min_val + 1)
            else:
                adjusted_values.append(max_val - value + 1)
        
        # Fix random choice with weights
        # Use numpy's choice function which supports weights
        index = np.random.choice(range(len(possible_moves)), p=np.array(adjusted_values)/sum(adjusted_values))
        return possible_moves[index]
    
    def best_child(self, c_param=None):
        """
        Better best child selection using UCB1 with adjustable exploration weight
        """
        if c_param is None:
            c_param = self.exploration_weight
            
        # terminal nodes
        for child in self.children:
            if child.state.is_terminal() and child.state.utility(self.player) == 1: # make sure we win !
                return child
        
        # No children case
        if not self.children:
            return None

        # Avoid array operations - use standard Python operations
        best_score = float('-inf')
        best_child = None
        
        # Calculate UCB for each child individually
        for child in self.children:
            if child.n() == 0:
                # Unexplored nodes get a very high priority
                score = float('inf')  # Python infinity, not NumPy
            else:
                # UCB formula using standard Python math
                exploitation = float(child.q()) / float(child.n())
                exploration = c_param * (2 * math.log(float(self.n())) / float(child.n())) ** 0.5
                
                # late game -> reduce exploration and focus on exploitation
                if self.state.turn >= LATE_GAME_LIMIT:  # Late game adjustment
                    exploration *= 0.5
                
                score = exploitation + exploration
            
            # Update best if this score is higher
            if best_child is None or score > best_score:
                best_score = score
                best_child = child
                
        return best_child
    
    def _tree_policy(self):
        """
        Better tree policy with time management
        """
        current_node = self
        depth = 0
        
        while not current_node.is_terminal_node() and depth < MAX_DEPTH:
            depth += 1
            
            # check time limit
            if self.is_time_up():
                return current_node
                
            # expand
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                next_node = current_node.best_child()
                if next_node is None:
                    return current_node
                current_node = next_node
                
        return current_node
    
    def best_action(self):
        """
        Better best action selection with time management
        """
        remaining_iterations = self.max_iterations
        iteration = 0
        
        while iteration < remaining_iterations:
            # check time limit
            if self.is_time_up():
                break
                
            try: # sometimes mcts errors so this prevent erros
                #DOCS: ()["MCTS Deep Dive.pdf"]
                # 1: Selection and expansion
                v = self._tree_policy()
                
                # 2: Simulation
                reward = v.rollout()
                
                # 3: Backpropagation
                v.backpropagate(reward)
                
                iteration += 1
            except Exception as e:
                print(f"Error in MCTS iteration {iteration}: {e}")
                # Check if this is the inhomogeneous array error
                if "inhomogeneous shape" in str(e):
                    # Convert possible array parameters to lists to avoid the error
                    if hasattr(self, '_untried_actions') and self._untried_actions:
                        self._untried_actions = list(self._untried_actions)
                    if self.children:
                        # Ensure any array operations on children will use lists
                        for child in self.children:
                            if hasattr(child, '_untried_actions') and child._untried_actions:
                                child._untried_actions = list(child._untried_actions)
                # skip this iteration but continue search
                iteration += 1
        
        # if we didn't complete any iterations, expand once
        if iteration == 0 and not self.children:
                self.expand()
        
        # if there is no children, return self
        if not self.children:
            print("No children during MCTS search")
            return self
            
        # get the best child (visit count) for final move choice
        # Safely convert to a list before np.array if needed
        visits_list = [child.n() for child in self.children]
        visits = np.array(visits_list)
        
        # check if we have any visited children
        if sum(visits) == 0:
            # if no visits, choose randomly among children
            best_child = np.random.choice(self.children)
        else:
            # Choose the child with the most visits
            best_child = self.children[np.argmax(visits)]
            
        return best_child
        # Strategy selection thresholds

class HybridAgent:
    def __init__(self, player):
        self.player = player
        
        # transposition table
        self.trans_table = TranspositionTable(max_size=MAX_SIZE_TRANSPOSITION)
        
        # performance tracking
        self.strategy_history = []
        self.time_usage_history = []
        self.last_execution_time = 0
        
        self.opening_book = None

        # predetermined moves
        if PREDETERMINED_START:
            self.opening_book = self._create_opening_book()
        
        
    def _create_opening_book(self):
        """
        Create an opening book with predetermined moves for the first few turns.
        Returns a dictionary mapping turn number to a FenixAction.
        """
        # first 10 moves
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
        """
        Determine the current game phase based on turn number and board state.
        """
        # check if we're in the opening phase (first few moves)
        if PREDETERMINED_START and state.turn in self.opening_book:
            return 'opening'
            
        # early game classification
        if state.turn < EARLY_GAME_THRESHOLD:
            return 'early'
            
        # Late game classification
        if state.turn >= LATE_GAME_THRESHOLD:
            return 'late'
            
        # Mid game classification
        if state.turn >= MID_GAME_THRESHOLD:
            return 'mid'
            
        
    def evaluate_complexity(self, state):
        """
        Evaluate the complexity of the position based on number of legal moves
        and board structure.
        """
        legal_moves = state.actions()
        num_moves = len(legal_moves)
        
        # complexity based on move count
        if num_moves <= LOW_COMPLEXITY_THRESHOLD:
            complexity = 'low'
        elif num_moves <= MID_COMPLEXITY_THRESHOLD:
            complexity = 'medium'
        else:
            complexity = 'high'
            
        # adjust complexity based on piece count
        piece_count = len(state.pieces)
        if piece_count <= 5:  # Very few pieces left
            complexity = 'low'  # Endgame tablebase territoryµ
        # not especially high complexity with lot of pieces !!!
            
        return complexity
        
    def evaluate_time_situation(self, remaining_time):
        """
        Evaluate the time situation.
        """
        if remaining_time <= CRITICAL_TIME_THRESHOLD:
            return 'critical'
        elif remaining_time <= LOW_TIME_THRESHOLD:
            return 'low'
        elif remaining_time <= MEDIUM_TIME_THRESHOLD:
            return 'medium'
        else:
            return 'high'
            
    def select_strategy(self, state, remaining_time):
        """
        Select the best strategy based on game phase, position complexity, and time remaining.
        
        Returns:
            dict: Strategy information including:
                - 'name': Strategy name ('opening', 'random', 'alpha_beta', or 'mcts')
                - 'params': Dict of parameters specific to the strategy
        """
        game_phase = self.determine_game_phase(state)
        complexity = self.evaluate_complexity(state)
        time_situation = self.evaluate_time_situation(remaining_time)
        
        
        # opening book moves take precedence if available
        if game_phase == 'opening':
            return {
                'name': 'opening',
                'params': {}
            }
            
        # critical time -> random -> optimize to put this code near call
        if time_situation == 'critical':
            return {
                'name': 'random',
                'params': {}
            }
            
        # low time situation -> use fast alpha-beta
        if time_situation == 'low':
            return {
                'name': 'alpha_beta',
                'params': {
                    'max_depth': MAX_DEPTH_A_B_LOW_TIME,
                    'time_limit': remaining_time * 0.8 # avoid timing out (allocate only 80% of time)
                }
            }
            
        # early game
        if game_phase == 'early':
            if complexity == 'low':
                # Low complexity early positions - deep alpha-beta
                return {
                    'name': 'alpha_beta',
                    'params': {
                        'max_depth': 7,
                        'time_limit': remaining_time * 0.8
                    }
                }
            else:
                # higher complexity -> mcts
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
                # high complexity mid game -> mcts !
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
                # lower complexity mid-game -> alpha-beta
                return {
                    'name': 'alpha_beta',
                    'params': {
                        'max_depth': 6,
                        'time_limit': remaining_time * 0.8
                    }
                }
        else:  # late game
            if complexity == 'low':
                # low complexity endgame - deep alpha-beta
                return {
                    'name': 'alpha_beta',
                    'params': {
                        'max_depth': 8,
                        'time_limit': remaining_time * 0.8
                    }
                }
            else:
                # higher complexity endgame - mcts exploitation
                return {
                    'name': 'mcts',
                    'params': {
                        'max_iterations': MAX_MCTS_ITERATIONS,
                        'rollout_depth': 4,
                        'time_limit': remaining_time * 0.8,
                        'exploration_weight': EXPLORATION_WEIGHT_END_GAME  # less exploration and more exploitation in endgame
                    }
                }
                
    def execute_opening_move(self, state):
        """
        Execute a move from the opening book.
        """
        # Get the predetermined move
        move_tuple = self.opening_book.get(state.turn)
        if not move_tuple:
            print(f"No opening move found for turn {state.turn}, falling back to Alpha-Beta")
            return self.execute_alpha_beta(state, {'max_depth': 4})
            
        start_pos, end_pos = move_tuple
        
        action = FenixAction(start_pos, end_pos, removed=frozenset())
        
        # verify validity
        legal_moves = state.actions()
        if action in legal_moves:
            return action
            
        # not valid move -> A-B
        print(f"Opening book move {action} not valid, falling back to Alpha-Beta")
        return self.execute_alpha_beta(state, {'max_depth': 4})
        
    def execute_random(self, state):
        """
        Execute a random move selection (for critical time situations).
        
        """
        legal_moves = state.actions()
        if not legal_moves:
            raise Exception("No legal moves available!")
            
        # select a random move
        action = np.random.choice(legal_moves)
        return action
        
    def execute_alpha_beta(self, state, params):
        """
        Execute Alpha-Beta search with the given parameters.
        """
        max_depth = params.get('max_depth', 4)
        time_limit = params.get('time_limit', None)
        alpha_beta = AlphaBeta(self.player, max_depth=max_depth)
        alpha_beta.transposition_table = self.trans_table
        
        start_time = time.time()
        
        try:
            action = alpha_beta.best_action(state)
            
            # Check if time limit was respected
            elapsed = time.time() - start_time
            if time_limit and elapsed > time_limit:
                print(f"Warning: Alpha-Beta exceeded time limit: {elapsed:.2f}s > {time_limit:.2f}s")
                # fall back to easier search
                alpha_beta.max_depth = MAX_DEPTH_A_B_LOW_TIME
                action = alpha_beta.best_action(state)
            
            return action
            
        except Exception as e:
            print(f"Error in Alpha-Beta search: {e}")
            return self.execute_random(state)
    
    def execute_mcts(self, state, params):
        """
        Execute MCTS search with the given parameters.
        """
        max_iterations = params.get('max_iterations', MAX_MCTS_ITERATIONS)
        rollout_depth = params.get('rollout_depth', 3)
        time_limit = params.get('time_limit', None)
        exploration_weight = params.get('exploration_weight', np.sqrt(2))

        # a-b for rollout
        alpha_beta = AlphaBeta(self.player, max_depth=rollout_depth)
        alpha_beta.transposition_table = self.trans_table
        
        try:
            # create mcts root
            mcts_root = BetterMCTSNode(
                state=state,
                player=self.player,
                alpha_beta=alpha_beta,
                max_iterations=max_iterations,
                rollout_depth=rollout_depth,
                time_limit=time_limit,
                exploration_weight=exploration_weight
            )
            
            # get best action
            selected_node = mcts_root.best_action()
            
            # extract action from the selected node
            action = selected_node.parent_action
            
            return action
            
        except Exception as e:
            print(f"MCTS failed: {e}")
            # Fall back to Alpha-Beta
            print("Falling back to Alpha-Beta due to unknown error")
            return self.execute_alpha_beta(state, {'max_depth': 3, 'time_limit': time_limit})
    
    def act(self, state, remaining_time):
        """
        Choose an action for the current state.
        """
        start_time = time.time()
        
        # legal moves
        legal_moves = state.actions()
        if not legal_moves:
            print("No legal moves available!")
            return None
        
        try:
            # select strategy
            strategy_info = self.select_strategy(state, remaining_time)
            strategy_name = strategy_info['name']
            strategy_params = strategy_info['params']
            
            # record strategy choice
            self.strategy_history.append({
                'turn': state.turn,
                'strategy': strategy_name,
                'time_remaining': remaining_time,
                'num_legal_moves': len(legal_moves),
                'num_pieces': len(state.pieces)
            })
            
            # print table stats before the search (only once every 5 moves to reduce output)
            if state.turn % 5 == 0 and DEBUG_MODE:
                print(f"\nTurn {state.turn} - Initial TT state:")
                self.trans_table.print_stats()
                
            # execute the selected strategy
            action = None
            if strategy_name == 'opening':
                action = self.execute_opening_move(state)
            elif strategy_name == 'random':
                action = self.execute_random(state)
            elif strategy_name == 'alpha_beta':
                action = self.execute_alpha_beta(state, strategy_params)
            elif strategy_name == 'mcts':
                action = self.execute_mcts(state, strategy_params)
            else:
                action = self.execute_random(state)
                
            # record execution time
            execution_time = time.time() - start_time
            self.last_execution_time = execution_time
            self.time_usage_history.append(execution_time)
            
            # Get detailed TT stats - only print once every 5 moves to avoid clutter
            if state.turn % 5 == 0 and DEBUG_MODE:
                print(f"\nTurn {state.turn} - After search TT state:")
                self.trans_table.print_stats()
            else:
                # Just print the hit rate every move
                if DEBUG_MODE:
                    hit_rate = self.trans_table.get_hit_rate()
                    print(f"Transposition table hit rate: {hit_rate:.2%} (Size: {len(self.trans_table.table)}/{self.trans_table.max_size})")
            
            return action
            
        except Exception as e:
            print(f"Error in act method: {e}")
            print("Using random move as ultimate fallback")
            return np.random.choice(legal_moves)
    
    def get_performance_summary(self):
        """
        Returns a summary of the agent's performance.
        """
        strategy_counts = {}
        for record in self.strategy_history:
            strategy = record['strategy']
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            
        avg_time = 0
        if self.time_usage_history:
            avg_time = sum(self.time_usage_history) / len(self.time_usage_history)
            
        return {
            'strategy_usage': strategy_counts,
            'avg_time_per_move': avg_time
        }
    
class Agent:
    def __init__(self, player):
        self.player = player

class monAgent(Agent):
    def __init__(self, player):
        self.player = player
        self.hybrid_agent = HybridAgent(self.player)

    def act(self, state, remaining_time):
        self.hybrid_agent.act(state, remaining_time)

    def __getattr__(self, attr):
        return getattr(self.hybrid_agent, attr)
    
