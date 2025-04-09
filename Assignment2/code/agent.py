#
# LINFO 1361 - Artificial Intelligence 
# Fenix game - April 2025
# Author: Arnaud Ullens, Quentin de Pierpont
# 

import time
from random import choice
from mcts import MonteCarloTreeSearchNode
from alpha_beta import AlphaBeta
import warnings
from consts import MAX_MCTS_ITERATIONS
from fenix import FenixAction
from hybrid_agent import BetterMCTSNode, HybridAgent

class Agent:
    def __init__(self, player):
        self.player = player
        self.hybrid_agent = HybridAgent(self.player)
    
    def __getattr__(self, attr):
        return getattr(self.hybrid_agent, attr)
