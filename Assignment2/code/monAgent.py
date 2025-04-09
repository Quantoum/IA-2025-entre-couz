#
# LINFO 1361 - Artificial Intelligence 
# Fenix game - April 2025
# Author: Arnaud Ullens, Quentin de Pierpont
# 


from random import *
from agent import Agent
#from mcts import MonteCarloTreeSearchNode
#from consts import MAX_MCTS_ITERATIONS
import warnings
import fenix


class monAgent(Agent):
    def __init__(self, player):
        self.player = player
        self.number_of_registered_moves = 2 # entre 0 et 5

    def act(self, state, remaining_time):
        """
            This function returns a FenixAction
        """
        # Check remaining time first and return a random legal action if time is exhausted
        if remaining_time <= 1:
            print("No more time.")
            return choice(state.actions())
        elif remaining_time <= 10:
            root = AlphaBeta(self.player, max_depth=4)
            return root.best_action(state)
        elif state.turn < 10 and self.number_of_registered_moves*2 > state.turn:
            return self.start_game(state)
        elif state.turn < 10:
            root = AlphaBeta(self.player, max_depth=4)
            return root.best_action(state)
        else:    
            try:
                # Create the MCTS node with the current state
                #root = MonteCarloTreeSearchNode(state=state, player=self.player, max_iterations=MAX_MCTS_ITERATIONS)
                root = AlphaBeta(self.player, max_depth=6)
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
        """
        registered_moves = {
            0: ((1,0),(0,0)), # red general
            1: ((5,7),(6,7)), # black general 
            2: ((0,1),(0,0)), # red king
            3: ((6,6),(6,7)), # black king
            4: ((1,1),(2,1)), # red general
            5: ((5,6),(4,6)), # black general
            6: ((0,2),(0,3)), # red general
            7: ((5,5),(4,5)), # black general
            8: ((1,2),(2,2)), # red general
            9: ((6,5),(6,4))  # black general
        }
        start, end = registered_moves.get(state.turn)
        return fenix.FenixAction(start, end, removed=frozenset())
    

class AlphaBeta:
    def __init__(self, player, max_depth=float('inf')):
        self.player = player
        self.max_depth = max_depth
        self.transposition_table = {}

    def alpha_beta_search(self, state):
        action = None
        if state.to_move() == self.player:
            _, action = self.max_value(state, -float('inf'), float('inf'), 0)
        else:
            _, action = self.min_value(state, -float('inf'), float('inf'), 0)
        return action

    def max_value(self, state, alpha, beta, depth):
        
        # Check if the state is already in the transposition table
        state_key = state._hash()
        if state_key in self.transposition_table:
            return self.transposition_table[state_key]
        
        if state.is_terminal() or depth >= self.max_depth:
            return self.heuristics(state), None
        
        #ordered_actions = sorted(state.actions(), key=lambda a: self.materialHeuristic(state.result(a)), reverse=False)
        
        value = -float('inf')
        action = None
        for a in state.actions():
            v, _ = self.min_value(state.result(a), alpha, beta, depth + 1)
            if v > value:
                value = v
                action = a
                alpha = max(alpha, value)
            if value >= beta:
                return value, action  # Beta cutoff
        
        # Store the value in the transposition table
        self.transposition_table[state_key] = (value, action)
        
        return value, action

    def min_value(self, state, alpha, beta, depth):
        
        # Check if the state is already in the transposition table
        state_key = state._hash()
        if state_key in self.transposition_table:
            return self.transposition_table[state_key]
        
        if state.is_terminal() or depth >= self.max_depth:
            return self.heuristics(state), None
        
        #ordered_actions = sorted(state.actions(), key=lambda a: self.materialHeuristic(state.result(a)), reverse=True)
        
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
            
        # Store the value in the transposition table
        self.transposition_table[state_key] = (value, action)
        
        return value, action

    def best_action(self, state): # for function names consistency
        return self.alpha_beta_search(state)
    
    def heuristics(self, state):
        if state.is_terminal():
            return 1000 * state.utility(self.player) # 1000 because it means the game is over
        
        score = 0
        score += 3 * self.materialHeuristic(state) # 10 because it forces the player to recreate a king
        score += 1 * self.positionalHeuristic(state)
        score += 1 * self.timeManaging(state)
        return score
    
    def materialHeuristic(self, state):
        # number of pieces on the board
        # at the start of the game, each player has 21 pieces
        score = 0
        for weight in state.pieces.values():
            score += weight
        
        return score
    
    '''
    #old function, may serve for the report and future characterization of the agent
    def positionalHeuristic(self, state):
        # the more pieces are in the border, the better
        on_border = 0
        for position, piece_value in state.pieces.items():
            # Check if this is your piece
            is_your_piece = False
            if self.player == 1 and piece_value > 0:  # Player 1 with positive pieces
                is_your_piece = True
            elif self.player == -1 and piece_value < 0:  # Player -1 with negative pieces
                is_your_piece = True
                
            if is_your_piece:
                # Check if piece is on border
                if position[0] == 0 or position[0] == 6:
                    on_border += 1
                if position[1] == 0 or position[1] == 7:
                    on_border += 1
                # Extra bonus if on the corner
                if position in [(0,0), (0,7), (6,0), (6,7)]:
                    on_border += 1
        return on_border'''
    
    def positionalHeuristic(self, state):
        # Precompute border positions
        border_positions = {(0,y) for y in range(8)} | {(6,y) for y in range(8)} | \
                        {(x,0) for x in range(7)} | {(x,7) for x in range(7)}
        
        on_border = 0
        for position, piece_value in state.pieces.items():
            if (self.player == 1 and piece_value > 0) or (self.player == -1 and piece_value < 0):
                if position in border_positions:
                    on_border += 1
        return on_border
    
    def timeManaging(self, state):
        # number of pieces on the board
        return 0
    