
import random
import time
import numpy as np
from environment import TerrainType, AntPerception, Direction
from ant import AntAction, AntStrategy

# good terminal input :
# python gui.py --strategy-file .\AntStrategy_concurrent.py --ants 1 --cell-size 3 --width 25 --height 25

class ConcurrentStrategy(AntStrategy):
    """
    A simple concurrent strategy for ants. The ants only counts on itself, 
    without any communication with other ants.

    

    to search for the food, we do a levy walk, which is a random walk with a
    power-law step length distribution. This means that the ants will take
    longer steps with a small probability, which allows them to explore a larger area.

    once the food is found, the ants will return to the colony in a straight line.

    --> need to store the coordinates of the food found, and the direction to the colony.
    """
    def __init__(self):
        """Initialize the strategy with last action tracking"""
        # Track the last action to alternate between movement and pheromone deposit
        self.ants_last_action = {}  # ant_id -> last_action
        self.next_movement_list = {}  # list of next movements to be done
        self.ant_positions = {i: (0, 0) for i in range(1, 11)}

    def decide_action(self, perception: AntPerception) -> AntAction:
        """Decide an action based on current perception"""

        # Get ant's ID to track its actions
        ant_id = perception.ant_id
        last_action = self.ants_last_action.get(ant_id, None)
    
        # # priority 1: find the food 
        # if not perception.has_food :
        #     if not perception.can_see_food(): 
        #         if self.next_movement_list == []:
        #             step_length = self.levy_walk()

        #     # if the ant is not carrying food and cannot see food, do a levy walk
        #     step_length = self.levy_walk()
        #     angle = random.uniform(0, 2 * np.pi)
        #     dx = step_length * np.cos(angle)
        #     dy = step_length * np.sin(angle)
        #     next_movement_list.append((dx, dy))


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
                        return self.go_forward_and_update_coordinate(perception)
        # If doesn't have food, try to move toward food if visible
        else:
            for pos, terrain in perception.visible_cells.items():
                if terrain == TerrainType.FOOD:
                    if pos[1] > 0:  # Food is ahead in some direction
                        return self.go_forward_and_update_coordinate(perception)

        # Random movement if no specific goal
        movement_choice = random.random()

        if movement_choice < 0.6:  # 60% chance to move forward
            return self.go_forward_and_update_coordinate(perception)
        elif movement_choice < 0.8:  # 20% chance to turn left
            return AntAction.TURN_LEFT
        else:  # 20% chance to turn right
            return AntAction.TURN_RIGHT

    def levy_walk(self, scale = 1):
        return scale * 1 / np.random.uniform(0, 1)
    
    def get_move(self):
        """Get the next move based on Levy walk"""
        step_length = self.levy_walk()
        angle = random.uniform(0, 2 * np.pi)
        dx = step_length * np.cos(angle)
        dy = step_length * np.sin(angle)
        return dx, dy
    
    def updtate_position(self, perception):
        if (len(perception.visible_cells) != 1 and not self.move_diagonal_on_border(perception)):
            # Update the position of the ant
            ant_id = perception.ant_id

            x, y = self.ant_positions[ant_id]
            dx, dy = perception.direction.get_delta(perception.direction)
            self.ant_positions[ant_id] = (x + dx, y + dy)
            print("Ant position: ", self.ant_positions[ant_id])
            # print("Ant direction: ", perception.direction)
            # print("Ant perception: ", perception.visible_cells)
            # wait for 1 second before moving again
            # time.sleep(1)

    def move_diagonal_on_border(self, perception):
        """Move diagonally on the border of the grid"""
        
    # [4,3,2] because when the and is on the border of the grid and goes diagonnally, it can see only 4,3,2 cells
    #     |                                         |
    #  ooo|A                                        |           
    #   oo|o                                        |                   
    #    o|o                                        |
    #     |o                                       A|ooo
    #     |                                        o|oo
    #     |________________________________________o|o
    #                                              o

        if(len(perception.visible_cells) in [4,3,2]): # means that the ant is on the border of the grid and trying to move out of the grid diagonally
            if perception.direction in [Direction.NORTHEAST, Direction.NORTHWEST, Direction.SOUTHEAST, Direction.SOUTHWEST]:
                return True
        return False
    
    def go_forward_and_update_coordinate(self, perception):
        self.updtate_position(perception)

        return AntAction.MOVE_FORWARD