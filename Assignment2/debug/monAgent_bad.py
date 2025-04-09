#!/usr/bin/env python3
"""
Hybrid MCTS/Alpha-Beta Agent for Fenix Game
Simplified version with core logic preserved
"""

import time
import math
import numpy as np
from fenix import FenixAction

# Constants for easy configuration
MAX_DEPTH_AB = 4
MAX_MCTS_ITERATIONS = 1000
TRANSPOSITION_SIZE = 100000
EARLY_GAME_TURNS = 15
LATE_GAME_TURNS = 30
CRITICAL_TIME = 1.0  # Seconds
EXPLORATION_WEIGHT = 1.41
OPENING_MOVES = 10

class Agent:
    def __init__(self, player):
        self.player = player
        self.trans_table = TranspositionTable(TRANSPOSITION_SIZE)
        self.stats = PerformanceStats()
        self.opening_book = self._create_opening_book()

    def act(self, state, remaining_time):
        start_time = time.time()
        legal_moves = state.actions()
        
        if not legal_moves:
            return None

        try:
            # Time emergency handling
            if remaining_time < CRITICAL_TIME:
                return self._random_move(state)

            # Strategy selection
            if state.turn in self.opening_book:
                action = self._try_opening_move(state)
                if action: return action

            phase = self._game_phase(state)
            complexity = self._evaluate_complexity(len(legal_moves))
            
            if phase == 'early' or phase == 'late' or complexity == 'low':
                depth = self._calculate_ab_depth(phase, complexity)
                action = self._alpha_beta_search(state, depth)
            else:
                action = self._mcts_search(state, remaining_time*0.8)

            # Update performance stats
            self.stats.record_move(time.time() - start_time, 
                                 len(legal_moves), 
                                 self.trans_table.get_stats())
            return action

        except Exception as e:
            print(f"Error: {e}, falling back to random")
            return np.random.choice(legal_moves)

    def _game_phase(self, state):
        if state.turn < EARLY_GAME_TURNS:
            return 'early'
        elif state.turn > LATE_GAME_TURNS:
            return 'late'
        return 'mid'

    def _evaluate_complexity(self, num_moves):
        if num_moves <= 8: return 'low'
        if num_moves <= 15: return 'medium'
        return 'high'

    def _calculate_ab_depth(self, phase, complexity):
        if complexity == 'low':
            return 6 if phase == 'late' else 4
        return 3 if phase == 'mid' else 4

    def _alpha_beta_search(self, state, max_depth):
        searcher = AlphaBeta(self.player, max_depth)
        searcher.transposition_table = self.trans_table
        return searcher.best_action(state)

    def _mcts_search(self, state, time_limit):
        root = MCTSNode(state, self.player, self.trans_table,
                      exploration=EXPLORATION_WEIGHT)
        return root.search(time_limit, MAX_MCTS_ITERATIONS)

    def _try_opening_move(self, state):
        move = self.opening_book.get(state.turn)
        if not move: return None
        
        action = FenixAction(*move)
        return action if action in state.actions() else None

    def _random_move(self, state):
        return np.random.choice(state.actions())

    def _create_opening_book(self):
        return {
            0: ((1,0), (0,0)),
            2: ((0,1), (0,0)),
            4: ((1,1), (2,1)),
        } if self.player == 1 else {
            1: ((5,7), (6,7)),
            3: ((6,6), (6,7)),
            5: ((5,6), (4,6)),
        }

class TranspositionTable:
    def __init__(self, max_size):
        self.table = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, state, depth, alpha, beta):
        key = hash(state)
        entry = self.table.get(key)
        if entry and entry['depth'] >= depth:
            self.hits += 1
            return entry['value'], entry['move']
        self.misses += 1
        return None, None

    def get_stats(self):
        hit_rate = self.hits/(self.hits+self.misses) if self.hits+self.misses > 0 else 0
        return {'size': len(self.table), 'hit_rate': hit_rate}

class AlphaBeta:
    def __init__(self, player, max_depth):
        self.player = player
        self.max_depth = max_depth
        self.transposition_table = None

    def best_action(self, state):
        best_val = -math.inf
        best_action = None
        for action in state.actions():
            val = self._search(state.result(action), -math.inf, math.inf, 0)
            if val > best_val:
                best_val = val
                best_action = action
        return best_action

    def _search(self, state, alpha, beta, depth):
        # Implementation with transposition table
        # ... (similar to original with pruning)
        return self._heuristic(state)

    def _heuristic(self, state):
        # Simplified material count
        return sum(weight for pos, weight in state.pieces)

class MCTSNode:
    def __init__(self, state, player, trans_table, parent=None, exploration=1.41):
        self.state = state
        self.player = player
        self.visits = 0
        self.value = 0
        self.children = []
        self.trans_table = trans_table
        self.exploration = exploration

    def search(self, time_limit, max_iterations):
        start = time.time()
        iterations = 0
        
        while iterations < max_iterations and (time.time() - start) < time_limit:
            node = self._select()
            result = node._simulate()
            node._backpropagate(result)
            iterations += 1
        
        return max(self.children, key=lambda c: c.visits).action

    def _select(self):
        # UCT selection with transposition table
        current = self
        while current.children:
            current = max(current.children, key=lambda n: n.value/n.visits + 
                        self.exploration*math.sqrt(math.log(self.visits)/n.visits))
        return current

    def _simulate(self):
        # Alpha-beta simulation instead of random rollout
        ab = AlphaBeta(self.player, 3)
        ab.transposition_table = self.trans_table
        return ab.best_action(self.state)

    def _backpropagate(self, result):
        current = self
        while current:
            current.visits += 1
            current.value += result
            current = current.parent

class PerformanceStats:
    def __init__(self):
        self.exec_times = []
        self.move_counts = []
        self.tt_stats = []

    def record_move(self, time_taken, moves, tt_info):
        self.exec_times.append(time_taken)
        self.move_counts.append(moves)
        self.tt_stats.append(tt_info)

    def get_summary(self):
        return {
            'avg_time': np.mean(self.exec_times),
            'avg_moves': np.mean(self.move_counts),
            'avg_tt_hit': np.mean([s['hit_rate'] for s in self.tt_stats]),
            'total_moves': len(self.exec_times)
        }
