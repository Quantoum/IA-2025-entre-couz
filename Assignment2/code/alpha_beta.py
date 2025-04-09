#
# LINFO 1361 - Artificial Intelligence 
# Fenix game - April 2025
# Author: Arnaud Ullens, Quentin de Pierpont
# 

from trans_table import TranspositionTable

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
    
        