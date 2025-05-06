from environment import TerrainType, AntPerception
from ant import AntAction, AntStrategy
from common import Direction
import random


class CollaborativeStrategy(AntStrategy):
    
    def __init__(self):
        """Initialize the strategy with last action tracking"""
        # Track the last action to alternate between movement and pheromone deposit
        self.ants_last_action = {}  # ant_id -> last_action
        self.exploration_rate = 0.05


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
        # Pheromone following or exploration
        # With some probability, explore instead of following pheromone
        if random.random() < self.exploration_rate:
            action = self._decide_movement(perception)
            self.ants_last_action[ant_id] = action
            return action

        # Determine which pheromone map to follow
        pheromone_map = None
        if perception.has_food and perception.home_pheromone:
            pheromone_map = perception.home_pheromone
        elif not perception.has_food and perception.food_pheromone:
            pheromone_map = perception.food_pheromone

        if pheromone_map:
            # Find direction of strongest pheromone
            pheromone_dir = self._find_strongest_pheromone(pheromone_map)
            if pheromone_dir is not None:
                current_dir = perception.direction.value
                # Move or turn toward pheromone direction
                if pheromone_dir == current_dir:
                    action = AntAction.MOVE_FORWARD
                elif (pheromone_dir - current_dir) % 8 <= 4:
                    action = AntAction.TURN_RIGHT
                else:
                    action = AntAction.TURN_LEFT
                self.ants_last_action[ant_id] = action
                return action

        # Otherwise, random movement
        action = self._decide_movement(perception)
        self.ants_last_action[ant_id] = action
        return action

    def _decide_movement(self, perception: AntPerception) -> AntAction:
        """Decide which direction to move based on current state"""

        # If has food, try to move toward colony if visible
        if perception.has_food and perception.can_see_colony():
            dir_to_colony = perception.get_colony_direction()
            if dir_to_colony is not None:
                if dir_to_colony == perception.direction.value:
                    return AntAction.MOVE_FORWARD
                elif (dir_to_colony - perception.direction.value) % 8 <= 4:
                    return AntAction.TURN_RIGHT
                else:
                    return AntAction.TURN_LEFT

        # If doesn't have food, try to move toward food if visible
        if not perception.has_food and perception.can_see_food():
            dir_to_food = perception.get_food_direction()
            if dir_to_food is not None:
                if dir_to_food == perception.direction.value:
                    return AntAction.MOVE_FORWARD
                elif (dir_to_food - perception.direction.value) % 8 <= 4:
                    return AntAction.TURN_RIGHT
                else:
                    return AntAction.TURN_LEFT

        # Random movement if no specific goal
        r = random.random()
        if r < 0.6:
            return AntAction.MOVE_FORWARD
        elif r < 0.8:
            return AntAction.TURN_LEFT
        else:
            return AntAction.TURN_RIGHT
        
    def _find_strongest_pheromone(self, pheromone_map: dict) -> int:
        """Find the direction of the strongest pheromone in the visible map"""
        # Select the position with maximum strength
        best_pos, best_strength = None, 0.0
        for pos, strength in pheromone_map.items():
            if strength > best_strength:
                best_strength = strength
                best_pos = pos
        if best_pos is None:
            return None
        dx, dy = best_pos
        return self._calculate_direction(dx, dy)

    def _calculate_direction(self, dx: int, dy: int) -> int:
        """Convert a delta (dx, dy) into the nearest discrete Direction value"""
        # Use same mapping as AntPerception
        if dx == 0 and dy < 0:
            return Direction.NORTH.value
        if dx > 0 and dy < 0:
            return Direction.NORTHEAST.value
        if dx > 0 and dy == 0:
            return Direction.EAST.value
        if dx > 0 and dy > 0:
            return Direction.SOUTHEAST.value
        if dx == 0 and dy > 0:
            return Direction.SOUTH.value
        if dx < 0 and dy > 0:
            return Direction.SOUTHWEST.value
        if dx < 0 and dy == 0:
            return Direction.WEST.value
        if dx < 0 and dy < 0:
            return Direction.NORTHWEST.value
        return Direction.NORTH.value
