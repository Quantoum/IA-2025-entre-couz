import random
import numpy as np
from environment import TerrainType, AntPerception, Direction
from ant import AntAction, AntStrategy

MAX_WALK_LENGTH = 40
MIN_WALK_LENGTH = 2
MU = 1.5
SCALE = 1
TURNS_TO_REFIND_FOOD_ZONE = 3

class ConcurrentStrategy(AntStrategy):
    def __init__(self):
        self.next_orientation_list = {}
        self.forward_motion_counter = {}
        self.ant_positions = {}
        self.was_blocked_on_something = {}
        self.known_food_zone = set()  # Shared memory for food locations
        self.just_arrived_at_previous_food_zone = {}
        self.following_wall = {}

    def decide_action(self, perception: AntPerception) -> AntAction:
        ant_id = perception.ant_id

        # Priority 1: Pick up food if standing on it
        if not perception.has_food and perception.visible_cells[(0, 0)] == TerrainType.FOOD:
            self.known_food_zone.add(self.ant_positions.get(ant_id))
            return AntAction.PICK_UP_FOOD

        # Priority 2: Drop food if at colony and carrying food
        if perception.has_food and perception.visible_cells[(0, 0)] == TerrainType.COLONY:
            return AntAction.DROP_FOOD

        # Priority 3: Search for food
        if not perception.has_food:
            if perception.can_see_food():
                return self.go_to_something(perception, TerrainType.FOOD)

            elif len(self.known_food_zone) > 0:
                current_pos = self.ant_positions.get(ant_id)
                closest = min(self.known_food_zone, key=lambda pos: (pos[0] - current_pos[0])**2 + (pos[1] - current_pos[1])**2)

                if current_pos == closest:
                    self.known_food_zone.remove(closest)
                    self.just_arrived_at_previous_food_zone[ant_id] = TURNS_TO_REFIND_FOOD_ZONE
                    return AntAction.TURN_RIGHT
                else:
                    return self.go_to_point(perception, *closest)

            elif self.just_arrived_at_previous_food_zone.get(ant_id, 0) > 0:
                self.just_arrived_at_previous_food_zone[ant_id] -= 1
                return AntAction.TURN_RIGHT

            else:
                return self.search_for_food(perception)

        # Priority 4: Return to colony with food
        if perception.has_food:
            return self.go_to_point(perception, 0, 0)

        return AntAction.TURN_LEFT  # Fallback action

    def go_to_something(self, perception, terrain_type=TerrainType.FOOD):
        for pos, terrain in perception.visible_cells.items():
            if terrain == terrain_type:
                target_x = pos[0] + self.ant_positions.get(perception.ant_id)[0]
                target_y = pos[1] + self.ant_positions.get(perception.ant_id)[1]
                return self.go_to_point(perception, target_x, target_y)

    def levy_walk(self, scale=1):
        u = np.random.uniform(0, 1)
        step_length = SCALE * (1.0 / (u ** (1.0 / (MU - 1))))
        return max(min(MAX_WALK_LENGTH, int(step_length)), MIN_WALK_LENGTH)

    def search_for_food(self, perception):
        ant_id = perception.ant_id

        if len(self.next_orientation_list.get(ant_id, [])) > 0:
            return self.next_orientation_list[ant_id].pop()

        elif self.forward_motion_counter.get(ant_id, 0) > 0:
            self.forward_motion_counter[ant_id] -= 1
            return self.go_forward_and_update_coordinate(perception)

        else:
            walk_length = self.levy_walk()
            self.forward_motion_counter[ant_id] = walk_length
            direction_choosen = random.choice([AntAction.TURN_LEFT, AntAction.TURN_RIGHT])
            direction_steps = random.randint(0, 4)
            self.next_orientation_list[ant_id] = [direction_choosen for _ in range(direction_steps)]
            return self.go_forward_and_update_coordinate(perception)

    def update_position(self, perception):
        if not self.blocked_on_something(perception):
            ant_id = perception.ant_id
            x, y = self.ant_positions.get(ant_id, (0, 0))
            dx, dy = perception.direction.get_delta(perception.direction)
            self.ant_positions[ant_id] = (x + dx, y + dy)

    def blocked_on_something(self, perception):
        dx, dy = perception.direction.get_delta(perception.direction)
        return perception.visible_cells.get((dx, dy), None) in (None, TerrainType.WALL)

    def detect_if_along_wall_right(self, perception):
        dx, dy = perception.direction.get_delta(Direction.get_right(perception.direction))
        return perception.visible_cells.get((dx, dy), None) in (None, TerrainType.WALL)

    def detect_if_along_wall_left(self, perception):
        dx, dy = perception.direction.get_delta(Direction.get_left(perception.direction))
        return perception.visible_cells.get((dx, dy), None) in (None, TerrainType.WALL)

    def go_forward_and_update_coordinate(self, perception):
        ant_id = perception.ant_id

        if self.blocked_on_something(perception):
            self.was_blocked_on_something[ant_id] = 1
            if not self.detect_if_along_wall_right(perception):
                return AntAction.TURN_RIGHT
            elif not self.detect_if_along_wall_left(perception):
                return AntAction.TURN_LEFT
            else:
                return AntAction.TURN_LEFT

        elif self.was_blocked_on_something.get(ant_id, 0) == 1:
            self.was_blocked_on_something[ant_id] = 0
            if not self.detect_if_along_wall_left(perception):
                return AntAction.TURN_LEFT
            elif self.detect_if_along_wall_right(perception):
                return AntAction.TURN_RIGHT

        self.was_blocked_on_something[ant_id] = 0
        self.update_position(perception)
        return AntAction.MOVE_FORWARD

    def go_to_point(self, perception, point_x, point_y):
        ant_id = perception.ant_id

        if ant_id not in self.following_wall:
            self.following_wall[ant_id] = False

        dx = point_x - self.ant_positions.get(ant_id)[0]
        dy = point_y - self.ant_positions.get(ant_id)[1]
        goal_dir = perception._get_direction_from_delta(dx, dy)
        delta_direction = (perception.direction.value - goal_dir) % 8

        if self.can_go_straight_to(perception, point_x, point_y):
            self.following_wall[ant_id] = False
            if delta_direction == 0:
                self.update_position(perception)
                return AntAction.MOVE_FORWARD
            elif delta_direction <= 4:
                return AntAction.TURN_LEFT
            else:
                return AntAction.TURN_RIGHT

        self.following_wall[ant_id] = True
        return self.wall_follow(perception)

    def can_go_straight_to(self, perception, target_x, target_y):
        dx = target_x - self.ant_positions.get(perception.ant_id)[0]
        dy = target_y - self.ant_positions.get(perception.ant_id)[1]
        goal_dir = perception._get_direction_from_delta(dx, dy)
        step_dx, step_dy = Direction.get_delta(goal_dir)
        return perception.visible_cells.get((step_dx, step_dy), None) not in (None, TerrainType.WALL)

    def wall_follow(self, perception):
        if not self.detect_if_along_wall_left(perception):
            return AntAction.TURN_LEFT
        elif not self.blocked_on_something(perception):
            self.update_position(perception)
            return AntAction.MOVE_FORWARD
        else:
            return AntAction.TURN_RIGHT
