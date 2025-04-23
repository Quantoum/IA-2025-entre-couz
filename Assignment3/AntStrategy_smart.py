from environment import TerrainType, AntPerception
from ant import AntAction, AntStrategy


class SmartStrategy(AntStrategy):
    
    def __init__(self):
        """Initialize the strategy with last action tracking"""
        # Track the last action to alternate between movement and pheromone deposit
        self.ants_last_action = {}  # ant_id -> last_action

    def decide_action(self, perception: AntPerception) -> AntAction:
        """Decide an action based on current perception"""
        pass