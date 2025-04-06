import random
import fenix

class AlphaBeta:
    def __init__(self, player, max_depth=float('inf')):
        self.player = player
        self.max_depth = max_depth

    def alpha_beta_search(self, state):
        action = None
        if state.to_move() == self.player:
            _, action = self.max_value(state, -float('inf'), float('inf'), 0)
        else:
            #_, action = self.min_value(state, -float('inf'), float('inf'), 0)
            pass
        return action

    def max_value(self, state, alpha, beta, depth):
        if state.is_terminal() or depth >= self.max_depth:
            return self.heuristics(state), None
        value = -float('inf')
        action = None
        for a in state.actions():
            v, _ = self.min_value(state.result(a), alpha, beta, depth + 1)
            if v > value:
                value = v
                action = a
                alpha = max(alpha, value)
            if value >= beta:
                return value, a  # Beta cutoff
        return value, action

    def min_value(self, state, alpha, beta, depth):
        if state.is_terminal() or depth >= self.max_depth:
            return self.heuristics(state), None
        value = float('inf')
        action = None
        for a in state.actions():
            v, _ = self.max_value(state.result(a), alpha, beta, depth + 1)
            if v < value:
                value = v
                action = a
                beta = min(beta, value)
            if value <= alpha:
                return value, action  # Alpha cutoff
        return value, action
    
    def best_action(self, state): # for function names consistency
        return self.alpha_beta_search(state)
    
    def heuristics(self, state):
        score = 0
        score += 10 * self.materialHeuristic(state) # 10 because it forces the player to recreate a king
        score += 1 * self.mobilityHeuristic(state)
        score += 1 * self.positionalHeuristic(state)
        score += 1 * self.pieceSafetyHeuristic(state)
        score += 1 * self.timeManaging(state)
        score += float('inf') * self.gameResult(state)
        return score
    
    def materialHeuristic(self, state):
        # number of pieces on the board
        # at the start of the game, each player has 21 pieces
        score = 0
        for piece in state.pieces.values():
            score += piece
        
        return score
        
    def mobilityHeuristic(self, state):
        # number of possible moves
        return 0
    
    def positionalHeuristic(self, state):
        # the more pieces are in the boarder, the better
        
        
        return 0
    
    def pieceSafetyHeuristic(self, state):
        # number of pieces on the board
        return 0
    
    def timeManaging(self, state):
        # number of pieces on the board
        return 0
    
    def gameResult(self, state):
        # gives the result of the game
        if state.is_terminal():
            return state.utility(self.player) - state.utility(-self.player)
        else:
            # game is not finished yet
            return 0
        