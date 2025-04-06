#
# LINFO 1361 - Artificial Intelligence 
# Fenix game - April 2025
# Author: Arnaud Ullens, Quentin de Pierpont
# 


from random import choice
from mcts import MonteCarloTreeSearchNode
from alpha_beta import AlphaBeta
import warnings
from consts import MAX_MCTS_ITERATIONS
from fenix import FenixAction
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
        # elif state.can_create_king == True:
        #     # create instantly a king
        #     return self.create_king(state)
        else:    
            try:
                # Create the MCTS node with the current state
                #root = MonteCarloTreeSearchNode(state=state, player=self.player, max_iterations=MAX_MCTS_ITERATIONS)
                root = AlphaBeta(self.player, max_depth=4)
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
    
    def create_king(self, state):
        # first, find all the generals
        # function not finished yet
        if state._count_generals() == 0:    
            return None
        number_of_preivous_generals = state._count_generals()
        for action in state.actions():
            if number_of_previous_generals + 1 == action._count_generals():
                return action
