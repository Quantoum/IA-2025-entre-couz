
import random
import time
import numpy as np
from environment import TerrainType, AntPerception, Direction
from ant import AntAction, AntStrategy

# ===============================================================================
# telepathy because the ants share the same variable for where the food is
# however, it is not yet implemented
# ===============================================================================

MAX_WALK_LENGTH = 30  # Maximum length of a walk in the levy walk
MIN_WALK_LENGTH = 2  # Minimum length of a walk in the levy walk
MU = 1.5  # Levy distribution parameter
SCALE = 1
TURNS_TO_REFIND_FOOD_ZONE = 3  # Number of turns to wait before searching for food again

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
        self.next_orientation_list = {}  
        self.forward_motion_counter = {} # list of next movements to do
        self.ant_positions = {}
        self.previously_blocked_on_something = {}
        self.known_food_zone = {}
        self.just_arrived_at_previous_food_zone = {}

    def decide_action(self, perception: AntPerception) -> AntAction:
        """Decide an action based on current perception"""

        # Priority 1: Pick up food if standing on it
        if (not perception.has_food and perception.visible_cells[(0, 0)] == TerrainType.FOOD):
            self.known_food_zone[perception.ant_id] = self.ant_positions.get(perception.ant_id)
            return AntAction.PICK_UP_FOOD

        # Priority 2: Drop food if at colony and carrying food
        if (perception.has_food and perception.visible_cells[(0,0)] == TerrainType.COLONY):
            return AntAction.DROP_FOOD

        # Priority 3: Search for food if not carrying food
        if not perception.has_food :

            # print("previous food zone: ", self.just_arrived_at_previous_food_zone.get(perception.ant_id, 0))
            if perception.can_see_food():
                action = self.go_to_something(perception, TerrainType.FOOD)  
            # if known food zone, go to it
            elif self.known_food_zone.get(perception.ant_id, None) is not None:
                # if we are at the position of the food, we remove it from the list
                if self.ant_positions.get(perception.ant_id) == self.known_food_zone[perception.ant_id]:
                    del self.known_food_zone[perception.ant_id]
                    action = AntAction.TURN_RIGHT # turn left to search for food again
                    self.just_arrived_at_previous_food_zone[perception.ant_id] = TURNS_TO_REFIND_FOOD_ZONE # wait x turns before searching for food again

                else:
                    target_x, target_y = self.known_food_zone[perception.ant_id]
                    action = self.go_to_point(perception, target_x, target_y)
            elif self.just_arrived_at_previous_food_zone.get(perception.ant_id, 0) > 0:
                action = AntAction.TURN_RIGHT
                self.just_arrived_at_previous_food_zone[perception.ant_id] -= 1
            else:
                action = self.search_for_food(perception)

        # Priority 4: Go to the colony if carrying food
        if perception.has_food :
            action = self.go_to_point(perception, 0, 0) # go to the colony

        return action
    
    def go_to_something(self, perception, terrain_type = TerrainType.FOOD):
        for pos, terrain in perception.visible_cells.items():
            if terrain == terrain_type:
                target_x = pos[0] + self.ant_positions.get(perception.ant_id)[0]
                target_y = pos[1] + self.ant_positions.get(perception.ant_id)[1]
                return self.go_to_point(perception, target_x, target_y)

    def levy_walk(self, scale = 1):
        u = np.random.uniform(0, 1)
        step_length = SCALE * (1.0 / (u ** (1.0 / (MU - 1))))
        walk_length = max(min(MAX_WALK_LENGTH, int(step_length)), MIN_WALK_LENGTH)
        return walk_length
                                                 
    def search_for_food(self, perception):
        ant_id = perception.ant_id

        if(len(self.next_orientation_list.get(ant_id, [])) != 0): # still orientation to do 
            # orientate
            # remove the first element of the list and return it
            next_orientation = self.next_orientation_list[ant_id].pop()
            return next_orientation
        
        elif(self.forward_motion_counter.get(ant_id, 0) != 0):
            # move
            # remove the first element of the list and return it
            self.forward_motion_counter[ant_id] -= 1
            action = self.go_forward_and_update_coordinate(perception)
            return action
        
        else: # all list are empty, we need to plan a levy walk
            walk_length = self.levy_walk()
            self.forward_motion_counter[ant_id] = walk_length # add the step length to the list of movements

            direction_choosen = random.choice([AntAction.TURN_LEFT, AntAction.TURN_RIGHT])
            direction_steps = random.randint(0, 4) # choose a random number between 0 and 4
            self.next_orientation_list[ant_id] = [direction_choosen for i in range(direction_steps)]
            return self.go_forward_and_update_coordinate(perception) # go forward and update the coordinate of the ant
    
    def update_position(self, perception):
        if (not self.blocked_on_something(perception)):
            # Update the position of the ant
            ant_id = perception.ant_id

            x, y = self.ant_positions.get(ant_id, (0, 0))  # Default to (0, 0) if not set
            dx, dy = perception.direction.get_delta(perception.direction)
            self.ant_positions[ant_id] = (x + dx, y + dy)
            if ant_id == 1:
                # print("Ant position: ", self.ant_positions[ant_id])
                pass

    def blocked_on_something(self, perception):
        dx, dy = perception.direction.get_delta(perception.direction)
        if(perception.visible_cells.get((dx, dy), None) in (None, TerrainType.WALL)):
            return True
        else:
            return False
    
    def go_forward_and_update_coordinate(self, perception):
        """ if we can go forward, we go forward and update the coordinate of the ant """

        # Check if the ant is blocked on something
        # If the ant is blocked, turn left
        if self.blocked_on_something(perception):
            self.previously_blocked_on_something[perception.ant_id] = True
            return AntAction.TURN_LEFT
        
        # avoid doing stupid turns around the map
        elif self.previously_blocked_on_something.get(perception.ant_id, False):
            self.previously_blocked_on_something[perception.ant_id] = False
            return AntAction.TURN_LEFT
        
        else:
            self.update_position(perception)
            return AntAction.MOVE_FORWARD

    def go_to_point(self, perception, point_x, point_y):
        """ go to a point in the map """
        # get the direction to the point
        dx = point_x - self.ant_positions.get(perception.ant_id)[0]
        dy = point_y - self.ant_positions.get(perception.ant_id)[1]
        # get the direction of the ant
        goal_direction = perception._get_direction_from_delta(dx, dy)
        delta_direction = (perception.direction.value - goal_direction) % 8

        # go forward if the direction is already good
        if delta_direction == 0:
            return self.go_forward_and_update_coordinate(perception)
        elif delta_direction <= 4:
            return AntAction.TURN_LEFT
        else:
            return AntAction.TURN_RIGHT