
#
# LINFO 1361 - Artificial Intelligence 
# Fenix game - April 2025
# Author: Arnaud Ullens, Quentin de Pierpont
# 

from hybrid_agent import HybridAgent
import monAgent

class Agent:
    def __init__(self, player):
        self.player = player
        self.hybrid_agent = monAgent(self.player)
    
    def __getattr__(self, attr):
        return getattr(self.hybrid_agent, attr)
