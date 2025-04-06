#
# LINFO 1361 - Artificial Intelligence 
# Fenix game - April 2025
# Author: Arnaud Ullens, Quentin de Pierpont
# 


from random import choice
#from mcts import MonteCarloTreeSearchNode
#from alpha_beta import AlphaBeta
import warnings
#from consts import MAX_MCTS_ITERATIONS
from fenix import FenixAction

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
        score += 1 * self.materialHeuristic(state)
        score += 1 * self.mobilityHeuristic(state)
        score += 1 * self.positionalHeuristic(state)
        score += 1 * self.pieceSafetyHeuristic(state)
        score += 1 * self.timeManaging(state)
        score += 100 * self.gameResult(state)
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
        # number of pieces on the board
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
class Agent:
    def __init__(self, player):
        self.player = player
        self.number_of_registered_moves = 0 # entre 0 et 5

    def act(self, state, remaining_time):
        """
            This function returns a FenixAction
        """
        # Check remaining time first and return a random legal action if time is exhausted
        if remaining_time <= 1:
            print("No more time.")
            return choice(state.actions())
        
        elif state.turn < 10 and self.number_of_registered_moves*2 > state.turn:
            return self.start_game(state)
        else:
            try:
                # Create the MCTS node with the current state
                #root = MonteCarloTreeSearchNode(state=state, player=self.player, max_iterations=MAX_MCTS_ITERATIONS)
                root = AlphaBeta(self.player, max_depth=2)
                # Get the best action using best_action()
                selected_node = root.best_action(state)
                
                # Extract and return the parent_action from the selected node
                return selected_node
            
            except RecursionError as e:
                warnings.warn(f"Recursion error during MCTS: {e}. Falling back to random action.")
                return choice(state.actions())
            except Exception as e:
                print(f"Error during MCTS: {e}")
                return choice(state.actions())
    
    def start_game(self, state):
        """
            This function is called at the start of the game
        """
        registered_moves = {
            0: ((0,0),(1,0)), # red general
            1: ((6,7),(5,7)), # black general 
            2: ((0,1),(1,1)), # red general
            3: ((6,6),(5,6)), # black general
            4: ((0,2),(1,2)), # red general
            5: ((6,5),(5,5)), # black general
            6: ((0,3),(0,4)), # red general
            7: ((6,4),(6,3)), # black general
            8: ((2,0),(1,0)), # red king
            9: ((4,7),(5,7))  # black king
        }
        
        """registered_moves = {
            0: ((0,0),(1,0)), # red general
            1: ((6,7),(5,7)), # black general 
            2: ((0,1),(1,1)), # red general
            3: ((6,6),(6,5)), # black general
            4: ((0,2),(1,2)), # red general
            5: ((5,6),(5,7)), # black general
            6: ((0,3),(0,4)), # red general
            7: ((6,4),(6,3)), # black general
            8: ((2,0),(1,0)), # red king
            9: ((4,7),(5,7))  # black king
        }"""
        
        start, end = registered_moves.get(state.turn)
        
        return FenixAction(start, end, removed=frozenset())