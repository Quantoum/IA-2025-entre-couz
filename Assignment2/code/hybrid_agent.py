#!/usr/bin/env python3
"""
Enhanced Hybrid Agent for Fenix Game
Integrates Monte Carlo Tree Search with Alpha-Beta pruning
"""

import time
import math
import os
import random
from typing import Optional, List, Dict, Tuple, Any

import numpy as np
from numpy import argmax, sqrt, log

from mcts import MonteCarloTreeSearchNode
from alpha_beta import AlphaBeta
from consts import EXPLO_CHILD_CONST, MAX_MCTS_ITERATIONS
from fenix import FenixAction

class EnhancedMCTSNode(MonteCarloTreeSearchNode):
    """
    Enhanced MCTS Node that uses Alpha-Beta pruning for rollouts and evaluations
    """
    def __init__(self, state, player, alpha_beta_instance, 
                 parent=None, parent_action=None, max_iterations=MAX_MCTS_ITERATIONS,
                 exploration_weight=sqrt(2), rollout_depth=3, time_limit=None):
        """
        Initialize an enhanced MCTS node
        
        Parameters:
            state: current game state
            player: player to move (1 or -1)
            alpha_beta_instance: instance of AlphaBeta to use for rollouts
            parent: parent node
            parent_action: action taken from parent to reach this node
            max_iterations: maximum number of iterations for MCTS
            exploration_weight: exploration weight for UCT
            rollout_depth: depth of Alpha-Beta search in rollouts
            time_limit: maximum time allowed for search (in seconds)
        """
        super().__init__(state, player, parent, parent_action, max_iterations)
        self.alpha_beta = alpha_beta_instance
        self.exploration_weight = exploration_weight
        self.rollout_depth = rollout_depth
        self.time_limit = time_limit
        self.start_time = time.time() if time_limit else None

    def is_time_up(self):
        """Check if the allocated time is up"""
        if self.time_limit is None or self.start_time is None:
            return False
        return (time.time() - self.start_time) >= self.time_limit
    def rollout(self):
        """
        Enhanced rollout that uses Alpha-Beta for short-term tactical evaluation
        """
        # If state is terminal, return the actual utility
        if self.state.is_terminal():
            result = self.state.utility(self.player)
            return result
            
        # For non-terminal states, use Alpha-Beta evaluation
        try:
            # Set a low depth for quick tactical evaluation
            self.alpha_beta.max_depth = self.rollout_depth
            
            # Do a quick Alpha-Beta search to evaluate the state
            ab_value = self.alpha_beta.heuristics(self.state)
            
            # Convert the AB evaluation to a normalized result between -1 and 1
            result = self._normalize_value(ab_value)
            return result
            
        except Exception as e:
            print(f"Warning: Error in enhanced rollout: {e}. Falling back to standard rollout.")
            # Fall back to standard rollout
            return super().rollout()
    
    def _normalize_value(self, value):
        """
        Convert an Alpha-Beta evaluation to a result between -1 and 1
        """
        # Assuming values are typically between -1000 and 1000
        # Scale down to -1 to 1 range
        MAX_VALUE = 1000.0
        return max(min(value / MAX_VALUE, 1.0), -1.0)
    
    def rollout_policy(self, possible_moves):
        """
        Enhanced rollout policy that uses heuristics to guide the selection
        """
        if not possible_moves:
            return None
            
        # If we're running out of time, use random selection
        if self.is_time_up() or len(possible_moves) > 20:
            return random.choice(possible_moves)
        
        # Evaluate each move using a lightweight heuristic
        move_values = []
        adjusted_values = []
        
        for move in possible_moves:
            try:
                next_state = self.state.result(move)
                # Use material and positional heuristics
                material_value = self.alpha_beta.materialHeuristic(next_state)
                positional_value = self.alpha_beta.positionalHeuristic(next_state)
                value = 3 * material_value + positional_value
                move_values.append(value)
            except Exception as e:
                print(f"Warning: Error evaluating move {move}: {e}")
                move_values.append(0)  # Neutral evaluation on error
        
        # If all evaluations failed, use random selection
        if not move_values or all(v == 0 for v in move_values):
            return random.choice(possible_moves)
            
        # Scale values to positive weights for selection
        min_val = min(move_values)
        max_val = max(move_values)
        
        # If all values are the same, choose randomly
        if max_val == min_val:
            return random.choice(possible_moves)
            
        # Scale values to positive weights for selection
        for value in move_values:
            if self.player == 1:  # Maximize for player 1
                adjusted_values.append(value - min_val + 1)
            else:  # Minimize for player -1
                adjusted_values.append(max_val - value + 1)
        
        # Use weights for weighted random selection
        total = sum(adjusted_values)
        probabilities = [v/total for v in adjusted_values]
    
    def best_child(self, c_param=None):
        """
        Enhanced best child selection using UCB1 with adjustable exploration weight
        """
        if c_param is None:
            c_param = self.exploration_weight
            
        # Check for terminal child nodes first
        for child in self.children:
            if child.state.is_terminal() and child.state.utility(self.player) == 1:
                return child
        
        # Compute UCB scores with optional adjustments
        choices_weights = []
        for c in self.children:
            if c.n() == 0:
                # Unexplored nodes get a high priority
                choices_weights.append(EXPLO_CHILD_CONST)
            else:
                # UCB1 formula with possible value transformation
                exploitation = c.q() / c.n()
                exploration = c_param * sqrt(2 * log(self.n()) / c.n())
                
                # For late game, reduce exploration and focus on exploitation
                if self.state.turn >= 30:  # Late game adjustment
                    exploration *= 0.5
                    
                weight = exploitation + exploration
                choices_weights.append(weight)
                
        # Pick the child with the highest score
        if not choices_weights:
            print("No children to select from in best_child!")
            return None
            
        return self.children[argmax(choices_weights)]
    
    def _tree_policy(self):
        """
        Enhanced tree policy with time management
        """
        current_node = self
        depth = 0
        max_depth = 50  # Prevent excessive depth
        
        while not current_node.is_terminal_node() and depth < max_depth:
            depth += 1
            
            # Check time limit
            if self.is_time_up():
                return current_node
                
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
        Enhanced best action selection with time management
        """
        remaining_iterations = self.max_iterations
        iteration = 0
        
        while iteration < remaining_iterations:
            # Check time limit
            if self.is_time_up():
                break
                
            try:
                # 1: Selection and expansion
                v = self._tree_policy()
                
                # 2: Simulation
                reward = v.rollout()
                
                # 3: Backpropagation
                v.backpropagate(reward)
                
                iteration += 1
            except Exception as e:
                print(f"Error in MCTS iteration {iteration}: {e}")
                # Skip this iteration on error but continue the search
                iteration += 1
        
        # If we didn't complete any iterations, expand once
        if iteration == 0 and not self.children:
            try:
                self.expand()
            except Exception as e:
                print(f"Error expanding root node: {e}")
        
        # If we have no children, return self (should not happen normally)
        if not self.children:
            print("No children generated during MCTS search!")
            return self
            
        # Get the best child based on visit count for the final move choice
        visits = np.array([child.n() for child in self.children])
        
        # Check if we have any visited children
        if np.sum(visits) == 0:
            # If no visits, choose randomly among children
            print("No child nodes have been visited during search!")
            best_child = random.choice(self.children)
        else:
            # Choose the child with the most visits
            best_child = self.children[np.argmax(visits)]
            
        # Strategy selection thresholds
    
    def __init__(self, player):
        self.player = player
        
        # Shared transposition table
        self.trans_table = TranspositionTable(max_size=MAX_SIZE_TRANSPOSITION)
        
        # Strategy selection thresholds
        
        # Complexity thresholds
        self.LOW_COMPLEXITY_THRESHOLD = 8   # <= 8 moves is low complexity
        self.MID_COMPLEXITY_THRESHOLD = 15  # <= 15 moves is medium complexity
        # > 15 moves is high complexity
        
        # Time thresholds
        self.CRITICAL_TIME_THRESHOLD = 1    # <= 1s is critical
        self.LOW_TIME_THRESHOLD = 5         # <= 5s is low
        self.MEDIUM_TIME_THRESHOLD = 20     # <= 20s is medium
        # > 20s is high
        
        # Performance tracking
        self.strategy_history = []
        self.time_usage_history = []
        self.last_execution_time = 0
        
        # Opening book (predetermined moves)
        self.opening_book = self._create_opening_book()
        
        
    def _create_opening_book(self):
        """
        Create an opening book with predetermined moves for the first few turns.
        Returns a dictionary mapping turn number to a FenixAction.
        """
        # Opening book structure
        # Key: turn number
        # Value: tuple of (start_pos, end_pos)
        
        # First 10 moves for both players
        if self.player == 1:  # Red player
            return {
                0: ((1,0),(0,0)),  # Move general towards the corner
                2: ((0,1),(0,0)),  # Move king on top of general in corner
                4: ((1,1),(2,1)),  # Move another general outward
                6: ((0,2),(0,3)),  # Move general to middle border
                8: ((1,2),(2,2)),  # Move another general outward
            }
        else:  # Black player
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
        
        Returns:
            str: 'opening', 'early', 'mid', or 'late'
        """
        # Check if we're in the opening phase (first few moves)
        if state.turn in self.opening_book:
            return 'opening'
            
        # Early game classification
        if state.turn < self.EARLY_GAME_THRESHOLD:
            return 'early'
            
        # Late game classification
        if state.turn >= self.LATE_GAME_THRESHOLD:
            return 'late'
            
        # Mid game classification
        if state.turn >= self.MID_GAME_THRESHOLD:
            return 'mid'
            
        # Default to early game if nothing else matches
        return 'early'
        
    def evaluate_complexity(self, state):
        """
        Evaluate the complexity of the position based on number of legal moves
        and board structure.
        
        Returns:
            str: 'low', 'medium', or 'high'
        """
        legal_moves = state.actions()
        num_moves = len(legal_moves)
        
        # Basic complexity based on move count
        if num_moves <= self.LOW_COMPLEXITY_THRESHOLD:
            complexity = 'low'
        elif num_moves <= self.MID_COMPLEXITY_THRESHOLD:
            complexity = 'medium'
        else:
            complexity = 'high'
            
        # Adjust complexity based on piece count
        piece_count = len(state.pieces)
        if piece_count <= 5:  # Very few pieces left
            complexity = 'low'  # Endgame tablebase territory
            
        return complexity
        
    def evaluate_time_situation(self, remaining_time):
        """
        Evaluate the time situation.
        
        Returns:
            str: 'critical', 'low', 'medium', or 'high'
        """
        if remaining_time <= self.CRITICAL_TIME_THRESHOLD:
            return 'critical'
        elif remaining_time <= self.LOW_TIME_THRESHOLD:
            return 'low'
        elif remaining_time <= self.MEDIUM_TIME_THRESHOLD:
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
        
        
        # Opening book moves take precedence if available
        if game_phase == 'opening':
            return {
                'name': 'opening',
                'params': {}
            }
            
        # Critical time situation - use fast approach
        if time_situation == 'critical':
            return {
                'name': 'random',
                'params': {}
            }
            
        # Low time situation - use fast alpha-beta
        if time_situation == 'low':
            return {
                'name': 'alpha_beta',
                'params': {
                    'max_depth': 3,
                    'time_limit': remaining_time * 0.8
                }
            }
            
        # Strategy selection based on game phase and complexity
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
                # Higher complexity early positions - balanced approach
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
                # High complexity mid-game - MCTS excels here
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
                # Lower complexity mid-game - alpha-beta
                return {
                    'name': 'alpha_beta',
                    'params': {
                        'max_depth': 6,
                        'time_limit': remaining_time * 0.8
                    }
                }
        else:  # late game
            if complexity == 'low':
                # Low complexity endgame - deep alpha-beta
                return {
                    'name': 'alpha_beta',
                    'params': {
                        'max_depth': 8,
                        'time_limit': remaining_time * 0.8
                    }
                }
            else:
                # Higher complexity endgame - MCTS with exploitative focus
                return {
                    'name': 'mcts',
                    'params': {
                        'max_iterations': MAX_MCTS_ITERATIONS,
                        'rollout_depth': 4,
                        'time_limit': remaining_time * 0.8,
                        'exploration_weight': 0.8  # Less exploration, more exploitation in endgame
                    }
                }
                
    def execute_opening_move(self, state):
        """
        Execute a move from the opening book.
        
        Returns:
            FenixAction: The selected opening book move
        """
        # Get the predetermined move
        move_tuple = self.opening_book.get(state.turn)
        if not move_tuple:
            print(f"No opening move found for turn {state.turn}, falling back to Alpha-Beta")
            return self.execute_alpha_beta(state, {'max_depth': 4})
            
        start_pos, end_pos = move_tuple
        
        # Create the action
        action = FenixAction(start_pos, end_pos, removed=frozenset())
        
        # Verify if the move is valid
        legal_moves = state.actions()
        if action in legal_moves:
            return action
            
        # If the move is not valid, fall back to Alpha-Beta
        print(f"Opening book move {action} not valid, falling back to Alpha-Beta")
        return self.execute_alpha_beta(state, {'max_depth': 4})
        
    def execute_random(self, state):
        """
        Execute a random move selection (for critical time situations).
        
        Returns:
            FenixAction: A randomly selected legal move
        """
        legal_moves = state.actions()
        if not legal_moves:
            print("No legal moves available!")
            return None
            
        # Select a random move
        action = random.choice(legal_moves)
        return action
        
    def execute_alpha_beta(self, state, params):
        """
        Execute Alpha-Beta search with the given parameters.
        
        Parameters:
            state: Current game state
            params: Dictionary of Alpha-Beta parameters
            
        Returns:
            FenixAction: The selected move
        """
        # Create Alpha-Beta instance
        alpha_beta = AlphaBeta(self.player, max_depth=max_depth)
        
        # Set up timer if needed
        start_time = time.time()
        
        try:
            action = alpha_beta.best_action(state)
            
            # Check if time limit was respected
            elapsed = time.time() - start_time
            if time_limit and elapsed > time_limit:
                print(f"Warning: Alpha-Beta exceeded time limit: {elapsed:.2f}s > {time_limit:.2f}s")
                
            
            
            # Update the transposition table from Alpha-Beta

            return action
            # Fall back to simpler search
            try:
                print("Falling back to simplified Alpha-Beta search")
                alpha_beta.max_depth = 2
                action = alpha_beta.best_action(state)
                return action
            except Exception as e2:
                print(f"Simplified Alpha-Beta search also failed: {e2}")
                # Ultimate fallback - random move
                return self.execute_random(state)
    
    def execute_mcts(self, state, params):
        """
        Execute MCTS search with the given parameters.
        
        Parameters:
            state: Current game state
            params: Dictionary of MCTS parameters
            
        Returns:
            FenixAction: The selected move
        """
        max_iterations = params.get('max_iterations', MAX_MCTS_ITERATIONS)
        rollout_depth = params.get('rollout_depth', 3)
        time_limit = params.get('time_limit', None)
        exploration_weight = params.get('exploration_weight', sqrt(2))

        
        # Create Alpha-Beta instance for rollouts
        alpha_beta = AlphaBeta(self.player, max_depth=rollout_depth)
        alpha_beta.transposition_table = self.trans_table.table
        
        try:
        alpha_beta = AlphaBeta(self.player, max_depth=rollout_depth)
        
        try:
                alpha_beta_instance=alpha_beta,
                trans_table=self.trans_table,
                max_iterations=max_iterations,
                player=self.player,
                alpha_beta_instance=alpha_beta,
                max_iterations=max_iterations,
                rollout_depth=rollout_depth,
                time_limit=time_limit,
                exploration_weight=exploration_weight
            selected_node = mcts_root.best_action()
            elapsed = time.time() - start_time
            
            # Extract the action from the selected node
            action = selected_node.parent_action
            
            return action
            
        except RecursionError as e:
            print(f"RecursionError in MCTS: {e}")
            # Fall back to Alpha-Beta
            print("Falling back to Alpha-Beta due to recursion error")
            return self.execute_alpha_beta(state, {'max_depth': 4, 'time_limit': time_limit})
            
        except Exception as e:
            print(f"MCTS failed: {e}")
            # Fall back to Alpha-Beta
            print("Falling back to Alpha-Beta due to general error")
            return self.execute_alpha_beta(state, {'max_depth': 3, 'time_limit': time_limit})
    
    def act(self, state, remaining_time):
        """
        Main method to choose an action for the current state with the given time constraint.
        
        Parameters:
            state: Current game state
            remaining_time: Time remaining in seconds
            
        Returns:
            FenixAction: The selected move
        """
        start_time = time.time()
        
        # Make sure we have legal moves
        legal_moves = state.actions()
        if not legal_moves:
            print("No legal moves available!")
            return None
        
        try:
            # Select strategy
            strategy_info = self.select_strategy(state, remaining_time)
            strategy_name = strategy_info['name']
            strategy_params = strategy_info['params']
            
            
            # Record strategy choice
            self.strategy_history.append({
                'turn': state.turn,
                'strategy': strategy_name,
                'time_remaining': remaining_time,
                'num_legal_moves': len(legal_moves),
                'num_pieces': len(state.pieces)
            })
            
            # Execute the selected strategy
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
                print(f"Unknown strategy: {strategy_name}")
                action = self.execute_random(state)
                
            # Record execution time
            execution_time = time.time() - start_time
            self.last_execution_time = execution_time
            self.time_usage_history.append(execution_time)
            
            # Print transposition table statistics
            hit_rate = self.trans_table.get_hit_rate()
            
            return action
            # Ultimate fallback - random move
            print("Using random move as ultimate fallback")
            return random.choice(legal_moves)
    
    def get_performance_summary(self):
        """
        Returns a summary of the agent's performance.
        
        Returns:
            dict: Performance summary
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
