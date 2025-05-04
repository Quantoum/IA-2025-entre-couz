from environment import TerrainType, AntPerception
from ant import AntAction, AntStrategy
import random


class CollaborativeStrategy(AntStrategy):
    
    def __init__(self):
        """Initialize the strategy with last action tracking"""
        # Track the last action to alternate between movement and pheromone deposit
        self.ants_last_action = {}  # ant_id -> last_action

    def decide_action(self, perception: AntPerception) -> AntAction:
        """Decide an action based on current perception"""

        """Decide an action based on current perception"""

        # Get ant's ID to track its actions
        ant_id = perception.ant_id
        last_action = self.ants_last_action.get(ant_id, None)

        # Priority 1: Pick up food if standing on it
        if (
            not perception.has_food
            and (0, 0) in perception.visible_cells
            and perception.visible_cells[(0, 0)] == TerrainType.FOOD
        ):
            self.ants_last_action[ant_id] = AntAction.PICK_UP_FOOD
            return AntAction.PICK_UP_FOOD

        # Priority 2: Drop food if at colony and carrying food
        if (
            perception.has_food
            and TerrainType.COLONY in perception.visible_cells.values()
        ):
            for pos, terrain in perception.visible_cells.items():
                if terrain == TerrainType.COLONY:
                    if pos == (0, 0):  # Directly on colony
                        self.ants_last_action[ant_id] = AntAction.DROP_FOOD
                        return AntAction.DROP_FOOD

        # Alternate between movement and dropping pheromones
        # If last action was not a pheromone drop, drop pheromone
        if last_action not in [
            AntAction.DEPOSIT_HOME_PHEROMONE,
            AntAction.DEPOSIT_FOOD_PHEROMONE,
        ]:
            if perception.has_food:
                self.ants_last_action[ant_id] = AntAction.DEPOSIT_FOOD_PHEROMONE
                return AntAction.DEPOSIT_FOOD_PHEROMONE
            else:
                self.ants_last_action[ant_id] = AntAction.DEPOSIT_HOME_PHEROMONE
                return AntAction.DEPOSIT_HOME_PHEROMONE
            
        # Suivre un phéromone si on est sur une case pheromone, ou si on le voit. 
        # -> problème : on voit plusieurs phéromones. 
        # Comment on fait ? -> on choisit le phéromone le plus puissant, si y'en a pas on bouge aléatoirement
        # On suit dans quel sens la ligne de phéromone -> en fct de l'intensité
        # X fois sur 10, on change de comportement (pour au cas ou trouver un meilleur chemin)

        # Otherwise, perform movement
        action = self._decide_movement(perception)
        self.ants_last_action[ant_id] = action
        return action

    def _decide_movement(self, perception: AntPerception) -> AntAction:
        """Decide which direction to move based on current state"""

        # If has food, try to move toward colony if visible
        if perception.has_food:
            for pos, terrain in perception.visible_cells.items():
                if terrain == TerrainType.COLONY:
                    if pos[1] > 0:  # Colony is ahead in some direction
                        return AntAction.MOVE_FORWARD
        # If doesn't have food, try to move toward food if visible
        else:
            for pos, terrain in perception.visible_cells.items():
                if terrain == TerrainType.FOOD:
                    if pos[1] > 0:  # Food is ahead in some direction
                        return AntAction.MOVE_FORWARD

        # Random movement if no specific goal
        movement_choice = random.random()

        if movement_choice < 0.6:  # 60% chance to move forward
            return AntAction.MOVE_FORWARD
        elif movement_choice < 0.8:  # 20% chance to turn left
            return AntAction.TURN_LEFT
        else:  # 20% chance to turn right
            return AntAction.TURN_RIGHT