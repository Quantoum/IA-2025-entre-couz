
import random
import time
import numpy as np
from environment import TerrainType, AntPerception, Direction
from ant import AntAction, AntStrategy

MAX_WALK_LENGTH = 40 # Maximum length of a walk in the levy walk
MIN_WALK_LENGTH = 4  # Minimum length of a walk in the levy walk
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
        self.known_food_zone = {}
        self.just_arrived_at_previous_food_zone = {}
        self.previous_wall_follow_type = {}
        self.explored_positions_while_wall_following = {}
        self.counter = {}


    def decide_action(self, perception: AntPerception) -> AntAction:
        """Decide an action based on current perception"""
        if perception.ant_id not in self.ant_positions:
            self.ant_positions[perception.ant_id] = (0, 0)  # Assuming ants always start at (0, 0)

        # Priority 1: Pick up food if standing on it
        if (not perception.has_food and perception.visible_cells[(0, 0)] == TerrainType.FOOD):
            self.known_food_zone[perception.ant_id] = self.ant_positions.get(perception.ant_id, (0,0))
            return AntAction.PICK_UP_FOOD

        # Priority 2: Drop food if at colony and carrying food
        if (perception.has_food and perception.visible_cells[(0,0)] == TerrainType.COLONY):
            return AntAction.DROP_FOOD

        # drop pheromone if counter is 0
        if self.counter.get(perception.ant_id, 0) == 0:
            if perception.has_food:
                self.counter[perception.ant_id] = 1
                return AntAction.DEPOSIT_FOOD_PHEROMONE
            else:
                self.counter[perception.ant_id] = 1
                return AntAction.DEPOSIT_HOME_PHEROMONE
        else:
            # next time we will not drop pheromone and perform an action
            self.counter[perception.ant_id] = 0

        # Priority 3: Search for food if not carrying food
        if not perception.has_food:
            if perception.can_see_food():
                food_x, food_y = self.get_coordinate_something(perception, TerrainType.FOOD)
                return self.go_to_point(perception, food_x, food_y) 

            # if known food zone, go to it
            elif self.known_food_zone.get(perception.ant_id, None) is not None:
                # if we are at the position of the food, we remove it from the list
                if self.ant_positions.get(perception.ant_id) == self.known_food_zone[perception.ant_id]:
                    del self.known_food_zone[perception.ant_id]
                    self.just_arrived_at_previous_food_zone[perception.ant_id] = TURNS_TO_REFIND_FOOD_ZONE # wait x turns before searching for food again
                    return AntAction.TURN_RIGHT # turn left to search for food again

                else:
                    target_x, target_y = self.known_food_zone[perception.ant_id]
                    return self.go_to_point(perception, target_x, target_y)
            # if we arrive at food zone but there is nothing left in our field of view
            elif self.just_arrived_at_previous_food_zone.get(perception.ant_id, 0) > 0:
                self.just_arrived_at_previous_food_zone[perception.ant_id] -= 1
                return AntAction.TURN_RIGHT
            
            elif self.find_strongest_pheromone_position(perception, perception.food_pheromone): # pheromone detection
                # Find direction of strongest pheromone
                dx, dy = self.find_strongest_pheromone_position(perception, perception.food_pheromone)
                return self.go_to_point(perception, dx, dy) # go to the strongest pheromone 


            # else we do a random walk to hope finding food
            else:
                return self.search_for_food(perception)

        # Priority 4: Go to the colony if carrying food
        if perception.has_food :
            if self.find_strongest_pheromone_position(perception, perception.home_pheromone) is not None:
                dx, dy = self.find_strongest_pheromone_position(perception, perception.home_pheromone)
                return self.go_to_point(perception, dx, dy) # go to the strongest pheromone
            else:
                return self.go_to_point(perception, 0, 0) # go to the colony

    def find_strongest_pheromone_position(self, perception, type):
        # Select the position with maximum strength
        best_pos, best_strength = None, 0.0
        for pos, strength in type.items():
            if strength > best_strength:
                best_strength = strength
                best_pos = pos
        if best_pos is None:
            return None
        dx, dy = best_pos
        target_x = dx + self.ant_positions.get(perception.ant_id)[0]
        target_y = dy + self.ant_positions.get(perception.ant_id)[1]
        return target_x, target_y
    
    def update_coordinate(self, perception):
        # Update the position of the ant
        ant_id = perception.ant_id

        x, y = self.ant_positions.get(ant_id, (0, 0))  # Default to (0, 0) if not set
        dx, dy = perception.direction.get_delta(perception.direction)
        self.ant_positions[ant_id] = (x + dx, y + dy)
        # if ant_id == 1:
        #     # print("Ant position: ", self.ant_positions[ant_id])
        #     pass

    def get_coordinate_something(self, perception, terrain_type):
        for pos, terrain in perception.visible_cells.items():
            if terrain == terrain_type:
                target_x = pos[0] + self.ant_positions.get(perception.ant_id)[0]
                target_y = pos[1] + self.ant_positions.get(perception.ant_id)[1]
                return target_x, target_y
            
    def detect_if_along_wall_left(self, perception):
        # the cell in the front right or front left
        dx1, dy1 = perception.direction.get_delta(Direction.get_left(perception.direction))
        if perception.visible_cells.get((dx1, dy1), None) in (None, TerrainType.WALL):
           return True               
        else:
            return False
        
    def detect_if_along_wall_right(self, perception):
        # the cell in the front right or front left
        dx1, dy1 = perception.direction.get_delta(Direction.get_right(perception.direction))
        if perception.visible_cells.get((dx1, dy1), None) in (None, TerrainType.WALL):
           return True
        else:
            return False
        
    def detect_if_along_walls(self, perception):
        # combinaison of the 2 functions wall left and wall right
        if self.detect_if_along_wall_left(perception) or self.detect_if_along_wall_right(perception):
            return True
        else: 
            return False

    def go_to_point(self, perception, point_x, point_y):

        ant_id = perception.ant_id

        dx = point_x - self.ant_positions.get(ant_id)[0]
        dy = point_y - self.ant_positions.get(ant_id)[1]
        goal_dir = perception._get_direction_from_delta(dx, dy)
        delta_direction = (perception.direction.value - goal_dir) % 8

        if self.ant_positions.get(ant_id) in self.explored_positions_while_wall_following.get(ant_id, []):
            if self.previous_wall_follow_type[perception.ant_id] == "left":
                self.previous_wall_follow_type[perception.ant_id] = "left"
                return self.follow_left_wall(perception)
            elif self.previous_wall_follow_type[perception.ant_id] == "right":
                self.previous_wall_follow_type[perception.ant_id] = "right"
                return self.follow_right_wall(perception)
            else:
                self.previous_wall_follow_type[perception.ant_id] = "left"
                return self.follow_left_wall(perception)

        if delta_direction == 0:
            self.previous_wall_follow_type[perception.ant_id] = None
            return self.try_go_forward_and_update_coordinate(perception)
        elif self.detect_if_along_walls(perception):
            if self.detect_if_along_wall_left(perception):  
                return self.follow_left_wall(perception)
            elif self.detect_if_along_wall_right(perception):
                return self.follow_right_wall(perception)
        elif self.previous_wall_follow_type.get(perception.ant_id, None) is not None:
            if self.previous_wall_follow_type[perception.ant_id] == "left":
                return self.follow_left_wall(perception)
            elif self.previous_wall_follow_type[perception.ant_id] == "right":
                return self.follow_right_wall(perception)
        elif delta_direction <= 4:
            self.previous_wall_follow_type[perception.ant_id] = None
            return AntAction.TURN_LEFT
        else:
            self.previous_wall_follow_type[perception.ant_id] = None
            return AntAction.TURN_RIGHT
    
    def follow_right_wall(self, perception):
        self.previous_wall_follow_type[perception.ant_id] = "right"
        self.explored_positions_while_wall_following[perception.ant_id].append(self.ant_positions.get(perception.ant_id))

        # si elle n'est pas bloquée va tout droit
        if self.blocked_on_something(perception):
            if self.detect_if_along_wall_right:
                return AntAction.TURN_LEFT
            else:
                return AntAction.TURN_RIGHT
        elif not self.detect_if_along_wall_right(perception):
            return AntAction.TURN_RIGHT
        else: 
            return self.try_go_forward_and_update_coordinate(perception)
        
    def follow_left_wall(self, perception):
        self.previous_wall_follow_type[perception.ant_id] = "left"
        self.explored_positions_while_wall_following[perception.ant_id].append(self.ant_positions.get(perception.ant_id))
        
        # si elle n'est pas bloquée va tout droit
        if self.blocked_on_something(perception):
            if self.detect_if_along_wall_left:
                return AntAction.TURN_RIGHT
            else:
                return AntAction.TURN_LEFT
        elif not self.detect_if_along_wall_left(perception):
            return AntAction.TURN_LEFT
        else: 
            return self.try_go_forward_and_update_coordinate(perception)

    def search_for_food(self, perception):
        ant_id = perception.ant_id

        # la fourmi numero 1 suit le mur de droite
        if ant_id == 1:
            return self.follow_right_wall(perception)

        if(len(self.next_orientation_list.get(ant_id, [])) != 0): # still orientation to do 
            # orientate
            # remove the first element of the list and return it
            next_orientation = self.next_orientation_list[ant_id].pop()
            return next_orientation
        
        elif(self.forward_motion_counter.get(ant_id, 0) != 0):
            # move
            
            # if we are not coincés entre deux murs, on évite de suivre les murs betement
            if not (self.detect_if_along_wall_left(perception) and self.detect_if_along_wall_right(perception)):
                if self.detect_if_along_wall_left(perception):
                    return AntAction.TURN_RIGHT
                elif self.detect_if_along_wall_right(perception):
                    return AntAction.TURN_LEFT
            # remove the first element of the list and return it
            self.forward_motion_counter[ant_id] -= 1
            action = self.try_go_forward_and_update_coordinate(perception)
            return action
        
        else: # all list are empty, we need to plan a levy walk
            walk_length = self.levy_walk()
            self.forward_motion_counter[ant_id] = walk_length # add the step length to the list of movements

            direction_choosen = random.choice([AntAction.TURN_LEFT, AntAction.TURN_RIGHT])
            direction_steps = random.randint(0, 4) # choose a random number between 0 and 4
            self.next_orientation_list[ant_id] = [direction_choosen for i in range(direction_steps)]
            return self.try_go_forward_and_update_coordinate(perception) # go forward and update the coordinate of the ant     

    def levy_walk(self, scale = 1):
        u = np.random.uniform(0, 1)
        step_length = SCALE * (1.0 / (u ** (1.0 / (MU - 1))))
        walk_length = max(min(MAX_WALK_LENGTH, int(step_length)), MIN_WALK_LENGTH)
        return walk_length
    
    def rebond_on_wall(self, perception):
        if self.detect_if_along_wall_left(perception):
            return AntAction.TURN_RIGHT
        elif self.detect_if_along_wall_right(perception):
            return AntAction.TURN_LEFT
        else:
            return AntAction.TURN_LEFT
    
    def try_go_forward_and_update_coordinate(self, perception):
        """ if we can go forward, we go forward and update the coordinate of the ant """

        # Check if the ant is blocked on something
        # If the ant is blocked turn approrpiately
        if self.blocked_on_something(perception):
            return self.rebond_on_wall(perception)
        #else go forward
        else:
            self.update_coordinate(perception)
            return AntAction.MOVE_FORWARD
        
    def blocked_on_something(self, perception):
        dx, dy = perception.direction.get_delta(perception.direction)
        if(perception.visible_cells.get((dx, dy), None) in (None, TerrainType.WALL)):
            return True
        else:
            return False