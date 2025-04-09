#!/usr/bin/env python3
"""
Hybrid Agent for Fenix Game
Integrates Monte Carlo Tree Search with Alpha-Beta pruning
"""

import time
import random
import numpy as np

from numpy import argmax, sqrt, log, array, sum

from mcts import MonteCarloTreeSearchNode
from alpha_beta import AlphaBeta
from fenix import FenixAction
from trans_table import TranspositionTable
from consts import *


class BetterMCTSNode(MonteCarloTreeSearchNode):
    def __init__(self, state, player, alpha_beta, 
                 parent=None, parent_action=None, max_iterations=MAX_MCTS_ITERATIONS,
                 exploration_weight=sqrt(2), rollout_depth=3, time_limit=None):
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
            return random.choice(possible_moves)
        
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
            return random.choice(possible_moves)
            
        # weighted randomness -> avoid local optima, exploration, compensate innacuracies in heuristic
        min_val = min(move_values)
        max_val = max(move_values)
        
        # all same values -> choose randomly -> optimized to avoid this ?
        if max_val == min_val:
            return random.choice(possible_moves)
            
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
        
        # compute UCB
        choices_weights = []
        for c in self.children:
            if c.n() == 0:
                # unexplored nodes get a high priority
                choices_weights.append(EXPLO_CHILD_CONST)
            else:
                # UCB1 formula
                exploitation = c.q() / c.n()
                exploration = c_param * sqrt(2 * log(self.n()) / c.n())
                
                # late game -> reduce exploration and focus on exploitation
                if self.state.turn >= LATE_GAME_LIMIT:  # Late game adjustment
                    exploration *= 0.5
                    
                weight = exploitation + exploration
                choices_weights.append(weight)
                
        # pick the child w/ highest score
        if not choices_weights:
            print("No children to select from in best_child!")
            return None
            
        return self.children[argmax(choices_weights)]
    
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
        visits = array([child.n() for child in self.children])
        
        # check if we have any visited children
        if sum(visits) == 0:
            # if no visits, choose randomly among children
            best_child = random.choice(self.children)
        else:
            # Choose the child with the most visits
            best_child = self.children[argmax(visits)]
            
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
        # Check if we're in the opening phase (first few moves)
        if PREDETERMINED_START and state.turn in self.opening_book:
            return 'opening'
            
        # Early game classification
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
        action = random.choice(legal_moves)
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
        exploration_weight = params.get('exploration_weight', sqrt(2))

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
            return random.choice(legal_moves)
    
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
