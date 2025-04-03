from random import choice
from mcts import MonteCarloTreeSearchNode
import warnings
from consts import MAX_MCTS_ITERATIONS

class Agent:
    def __init__(self, player=None):
        self.player = player

    def act(self, state, remaining_time):
        """
            This function returns a FenixAction
        """
        # Check remaining time first and return a random legal action if time is exhausted
        if remaining_time <= 0:
            print("No more time.")
            return choice(state.actions())
            
        try:
            # Create the MCTS node with the current state
            root = MonteCarloTreeSearchNode(state=state, player=self.player, max_iterations=MAX_MCTS_ITERATIONS)
            
            # Get the best action using best_action()
            selected_node = root.best_action()
            
            # Extract and return the parent_action from the selected node
            return selected_node.parent_action
        
        except RecursionError as e:
            warnings.warn(f"Recursion error during MCTS: {e}. Falling back to random action.")
            return choice(state.actions())
        except Exception as e:
            print(f"Error during MCTS: {e}")
            return choice(state.actions())