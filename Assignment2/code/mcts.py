
#
# LINFO 1361 - Artificial Intelligence 
# Fenix game - April 2025
# Author: Arnaud Ullens, Quentin de Pierpont
# 

from numpy import argmax, sqrt, log
from random import choice
from consts import EXPLO_CHILD_CONST

class MonteCarloTreeSearchNode():
    def __init__(self, state, player, parent=None, parent_action=None, max_iterations=100):
        self.state = state # board state
        self.player = player # player that has to move
        self.parent = parent # None for root node, equals to the node it's derived from/
        self.parent_action = parent_action # None for root node, equals action which it's parent carried out.
        self.max_iterations = max_iterations
        self.children = [] # all possible actions from the current node
        self._number_of_visits = 0 # number of time a current node is visited (back-propagation)
        self._results = {}
        self._results[1] = 0
        self._results[0] = 0
        self._results[-1] = 0
        self._untried_actions = None
        self._untried_actions = self.untried_actions() # list of all possible actions
    
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
        return result
    
    def backpropagate(self, result):
        """
            This function computes all the statistics for the nodes that are updated. Until the parent
            node is reached, the number of visits is incremented. It increment by 1 the number of wins
            (or losses) depending on the result. Recursive.
        """
        self._number_of_visits += 1
        self._results[result] += 1
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
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node
    
    def best_action(self):
        """
            This function compute the best possible move.
            @return: the best action
        """
        for i in range(self.max_iterations):
            # 1 : Selection and expansion (find a node to simulate from)
            v = self._tree_policy()
            # 2 : Simulation (play out to terminal state)
            reward = v.rollout()
            # 3 : Backpropagation (update statistics up the tree)
            v.backpropagate(reward)
        best = self.best_child(c_param=sqrt(2))
        return best
    
