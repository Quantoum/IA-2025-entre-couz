#
# LINFO 1361 - Artificial Intelligence 
# Fenix game - April 2025
# Author: Arnaud Ullens, Quentin de Pierpont
# 

# Import the TranspositionTable class to use its methods and constants
from trans_table import TranspositionTable

class AlphaBeta:
    def __init__(self, player, max_depth=float('inf')):
        self.player = player
        self.max_depth = max_depth
        # self.transposition_table should be an instance of TranspositionTable, set by the agent
        self.transposition_table = None 

    def alpha_beta_search(self, state):
        # Ensure the transposition table is assigned before starting
        if self.transposition_table is None:
            raise ValueError("Transposition table not set for AlphaBeta agent.")
            
        # Determine the initial call based on the current player
        # We don't store the action from the top level, only the value.
        # The best action is determined by iterating through the first level moves.
        
        best_action = None
        if state.to_move() == self.player:
            # Player is MAX player
            best_value = -float('inf')
            for action in state.actions():
                value, _ = self.min_value(state.result(action), -float('inf'), float('inf'), 0)
                if value > best_value:
                    best_value = value
                    best_action = action
        else:
            # Player is MIN player (opponent's turn)
            best_value = float('inf')
            for action in state.actions():
                value, _ = self.max_value(state.result(action), -float('inf'), float('inf'), 0)
                if value < best_value:
                    best_value = value
                    best_action = action
                    
        return best_action

    def max_value(self, state, alpha, beta, depth):
        # Early cutoff for terminal states or max depth
        if state.is_terminal() or depth >= self.max_depth:
            return self.heuristics(state), None
        
        # --- Transposition Table Lookup --- 
        original_alpha = alpha  # Store original alpha for TT storage flag
        tt_value, tt_move = self.transposition_table.get(state, depth, alpha, beta)
        if tt_value is not None:
            return tt_value, tt_move  # Return the stored move as well
        # --- End TT Lookup ---
        
        value = -float('inf')
        best_action_for_node = None
        
        # Consider move ordering here using tt_move if available
        actions = state.actions()
        # simple move ordering: try stored best move first
        if tt_move is not None and tt_move in actions:
            actions.remove(tt_move)
            actions.insert(0, tt_move)
             
        for action in actions:
            v, _ = self.min_value(state.result(action), alpha, beta, depth + 1)
            if v > value:
                value = v
                best_action_for_node = action  # Track the best action found at this node
                alpha = max(alpha, value)  # Alpha update
                
            # Beta cutoff check
            if value >= beta:
                # --- Transposition Table Store (Beta Cutoff) --- 
                self.transposition_table.put(state, depth, value, TranspositionTable.LOWER_BOUND, best_action_for_node)
                # --- End TT Store ---
                return value, best_action_for_node  # Beta cutoff
        
        # --- Transposition Table Store (Exact or Upper Bound) ---
        if value <= original_alpha:  # Failed low (Upper Bound for this node)
            flag = TranspositionTable.UPPER_BOUND
        else:  # Found an exact value within the alpha-beta window
            flag = TranspositionTable.EXACT
        self.transposition_table.put(state, depth, value, flag, best_action_for_node)
        # --- End TT Store ---
        
        return value, best_action_for_node

    def min_value(self, state, alpha, beta, depth):
        # Early cutoff for terminal states or max depth
        if state.is_terminal() or depth >= self.max_depth:
            return self.heuristics(state), None
            
        # --- Transposition Table Lookup --- 
        original_beta = beta  # Store original beta for TT storage flag
        tt_value, tt_move = self.transposition_table.get(state, depth, alpha, beta)
        if tt_value is not None:
            return tt_value, tt_move  # Return stored move
        # --- End TT Lookup ---
        
        value = float('inf')
        best_action_for_node = None
        
        # Consider move ordering here using tt_move if available
        actions = state.actions()
        # simple move ordering: try stored best move first
        if tt_move is not None and tt_move in actions:
            actions.remove(tt_move)
            actions.insert(0, tt_move)
             
        for action in actions:
            v, _ = self.max_value(state.result(action), alpha, beta, depth + 1)
            if v < value:
                value = v
                best_action_for_node = action  # Track best action
                beta = min(beta, value)  # Beta update
                
            # Alpha cutoff check
            if value <= alpha:
                # --- Transposition Table Store (Alpha Cutoff) --- 
                self.transposition_table.put(state, depth, value, TranspositionTable.UPPER_BOUND, best_action_for_node)
                # --- End TT Store ---
                return value, best_action_for_node  # Alpha cutoff
            
        # --- Transposition Table Store (Exact or Lower Bound) ---
        if value >= original_beta:  # Failed high (Lower Bound for this node)
            flag = TranspositionTable.LOWER_BOUND
        else:  # Found an exact value within the alpha-beta window
            flag = TranspositionTable.EXACT
        self.transposition_table.put(state, depth, value, flag, best_action_for_node)
        # --- End TT Store ---
        
        return value, best_action_for_node

    def best_action(self, state):
        """Entry point for finding the best action."""
        return self.alpha_beta_search(state)
    
    def heuristics(self, state):
        if state.is_terminal():
            # Use a large value for terminal states
            utility = state.utility(self.player)
            if utility == 1: return 10000 # Win
            if utility == -1: return -10000 # Loss
            return 0 # Draw
        
        score = 0
        # Material heuristic - weighted sum of piece values
        # Positional heuristic - encourage pieces on border
        score += 3 * self.materialHeuristic(state) 
        score += 1 * self.positionalHeuristic(state)
        # score += 1 * self.timeManaging(state) # timeManaging seems unused
        return score
    
    def materialHeuristic(self, state):
        """Calculates material balance based on piece weights."""
        score = 0
        # state.pieces is a dictionary where:
        # - keys are positions (tuples)
        # - values are piece weights (integers)
        for position, weight in state.pieces.items():
            # Add the weight (positive for player 1, negative for player -1)
            score += weight
        
        # Ensure the score is relative to our player's perspective
        if self.player == -1:
            score = -score  # Negate if we're player -1
        
        return score
    
    def positionalHeuristic(self, state):
        """Simple heuristic: counts pieces on the border."""
        # Precompute border positions
        border_positions = {(0,y) for y in range(8)} | {(6,y) for y in range(8)} | \
                        {(x,0) for x in range(7)} | {(x,7) for x in range(7)}
        
        on_border = 0
        opponent_on_border = 0
        
        # state.pieces is a dictionary where:
        # - keys are positions (tuples)
        # - values are piece weights (integers)
        for position, weight in state.pieces.items():
            if position in border_positions:
                # If the weight is positive and we're player 1, it's our piece
                # If the weight is negative and we're player -1, it's our piece
                is_our_piece = (weight > 0 and self.player == 1) or (weight < 0 and self.player == -1)
                
                if is_our_piece:
                    on_border += 1
                else:
                    opponent_on_border += 1
                    
        # Return difference to penalize opponent's border control
        return on_border - opponent_on_border 
    
    # timeManaging seems placeholder, returning 0
    # def timeManaging(self, state):
    #     return 0
        
        