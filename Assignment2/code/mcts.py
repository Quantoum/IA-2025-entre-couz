
"""
Created at 22:15 01-04-25.

Author : Arnaud Ullens

Modification History :
    - 02-04-25 : Arnaud Ullens
"""

from numpy import argmax, sqrt, log
from random import choice
from collections import defaultdict
import logging
import os
import datetime
from consts import EXPLO_CHILD_CONST

# Create logs directory if it doesn't exist
logs_dir = "logs"
try: 
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
        print(f"Created logs directory: {logs_dir}")
except OSError as e:
    print(f"Error creating logs directory: {e}")
    # Fall back to current directory if unable to create logs directory
    logs_dir = "."

# Generate timestamp for unique log filenames
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Create a custom logger
logger = logging.getLogger('mcts')
logger.setLevel(logging.DEBUG)

# Create handlers with timestamped filenames
debug_log_file = os.path.join(logs_dir, f"mcts_debug_{timestamp}.log")
debug_handler = logging.FileHandler(debug_log_file)
debug_handler.setLevel(logging.DEBUG)

info_log_file = os.path.join(logs_dir, f"mcts_info_{timestamp}.log")
info_handler = logging.FileHandler(info_log_file)
info_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatters and add them to handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
debug_handler.setFormatter(formatter)
info_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(debug_handler)
logger.addHandler(info_handler)
logger.addHandler(console_handler)

logger.info(f"MCTS logger initialized with log files: {debug_log_file} and {info_log_file}")

class MonteCarloTreeSearchNode():
    def __init__(self, state, player, parent=None, parent_action=None, max_iterations=100):
        self.state = state # board state
        self.player = player # player that has to move
        self.parent = parent # None for root node, equals to the node it's derived from/
        self.parent_action = parent_action # None for root node, equals action which it's parent carried out.
        self.max_iterations = max_iterations
        self.children = [] # all possible actions from the current node
        self._number_of_visits = 0 # number of time a current node is visited (back-propagation)
        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[0] = 0
        self._results[-1] = 0
        self._untried_actions = None
        self._untried_actions = self.untried_actions() # list of all possible actions
        logger.debug(f"Created node with state {self.state} player {self.player} and parent action {parent_action}")
    
    def untried_actions(self):
        """
            This function gets the list of untried actions from a given state.
            @return : list of untried actions
        """
        self._untried_actions = self.state.actions()
        return self._untried_actions
    
    def q(self):
        """
            This function computes the difference between wins and loses.
            @return : wins - loses
        """
        wins = self._results[1]
        loses = self._results[-1]
        return wins - loses
    
    def n(self):
        """
            This function computes the number of times each node is visited.
            @return: the number of visits
        """
        return self._number_of_visits
    
    def expand(self):
        """
            This function generates the next state (from the present state) depending
            on the action that has been carried out. 
            @return: The child node, a.k.a the next state.
        """
        action = self._untried_actions.pop()
        next_state = self.state.result(action)
        child_node = MonteCarloTreeSearchNode(next_state, player=self.player, parent=self, parent_action=action)
        self.children.append(child_node)
        logger.debug(f"Expanded node with action {action}, created child : {child_node}")
        logger.info(f"Expanded with action {action}")
        return child_node
    
    def is_terminal_node(self):
        """
            This function checks wheter the state is final or not.
            @return: [bool] current not is terminal.
        """
        return self.state.is_terminal()
    
    def rollout(self):
        """
            This function generates an entire game from the current state (until there is an outcome).
            @return: the outcome of the game. (win=1, loss=-1, tie=0) -> depending on policy."""
        current_rollout_state = self.state

        while not current_rollout_state.is_terminal():
            possible_moves = current_rollout_state.actions()
            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.result(action)
        result = current_rollout_state.utility(self.player)
        logger.debug(f"Completed rollout with result: {result}")
        return result
    
    def backpropagate(self, result):
        """
            This function computes all the statistics for the nodes that are updated. Until the parent
            node is reached, the number of visits is incremented. It increment by 1 the number of wins
            (or losses) depending on the result. Recursive.
        """
        self._number_of_visits += 1
        self._results[result] += 1
        logger.debug(f"Backpropagating result {result} to node. Visits: {self._number_of_visits}")
        if self.parent:
            self.parent.backpropagate(result)
    
    def is_fully_expanded(self):
        """
            This function is checking whether the node is fully expanded or not. All the actions are
            popped out of _untried_actions one by one. When it's empty, it's fully expanded.
            @return: whether the node is fully expanded or not.
        """
        return len(self._untried_actions) == 0
    
    def best_child(self, c_param=0.1):
        """
            This function selects the best child out of the children (once fully expanded).
            See the formula explanation in the associated documentation.
            @return: The best child.
        """
        choices_weights = []
        for c in self.children:
            if c.n() == 0:
                # unexplored nodes; ensure they are explored by adding large value
                choices_weights.append(EXPLO_CHILD_CONST)
            else:
                # UCT for explored nodes
                exploitation = c.q() / c.n()
                exploration = c_param * sqrt(2 * log(self.n()) / c.n())
                weight = exploitation + exploration
                logger.debug(f"Child weight calculation: exploitation={exploitation:.4f}, exploration={exploration:.4f}, total={weight:.4f}")
                choices_weights.append(weight)
        return self.children[argmax(choices_weights)]
    
    def rollout_policy(self, possible_moves):
        """
            This function selects a random move in all possible moves. It's an "example payout".
            Can be improved by heuristic, and/or Deep RL !
            @return: the selected move.
        """
        return choice(possible_moves)

    def _tree_policy(self):
        """
            This function selects the node to run rollout. Returns the best child, or expand the
            node.
            @return: the node, expanded, or the best child.
        """
        current_node = self
        depth = 0
        max_depth = 50
        while not current_node.is_terminal_node() and depth < max_depth:
            depth += 1
            logger.info(f"Tree policy at depth: {depth}")
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
                logger.debug(f"Selected best child, moving to depth {depth}")
        logger.info(f"Tree policy complete at depth {depth}")
        return current_node
    
    def best_action(self):
        """
            This function compute the best possible move.
            @return: the best action
        """
        logger.info(f"Starting MCTS with {self.max_iterations} iterations")
        for i in range(self.max_iterations):
            if i % 10 == 0:
                logger.info(f"MCTS iteration {i}/{self.max_iterations}")
            # 1 : Selection and expansion (find a node to simulate from)
            v = self._tree_policy()
            # 2 : Simulation (play out to terminal state)
            reward = v.rollout()
            logger.debug(f"Rollout result for iteration {i}: {reward}")
            # 3 : Backpropagation (update statistics up the tree)
            v.backpropagate(reward)
        logger.info(f"MCTS complete, selecting best child from {len(self.children)} options")
        best = self.best_child(c_param=sqrt(2))
        logger.info(f"Selected best action: {best.parent_action}")
        return best
    
