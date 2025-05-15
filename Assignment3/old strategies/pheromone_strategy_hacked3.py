import random

from ant import AntStrategy
from common import AntAction, AntPerception, Direction, TerrainType

class ConcurrentStrategy(AntStrategy):
    """
    A pheromone-based ant strategy that uses pheromone trails for food collection:

    Behavior:
    1. Exploration phase: Ants randomly explore the environment looking for food
    2. Food collection: Once food is found, ants follow home pheromones back to colony
    3. Return to food: Ants at colony follow food pheromones to return to food sources

    Pheromones are used as a communication mechanism between ants:
    - Food pheromones: Deposited when carrying food, help other ants find food
    - Home pheromones: Deposited when not carrying food, help ants find way back to colony
    """

    def __init__(self):
        """Initialize the pheromone-based strategy"""
        self.pheromone_threshold = 0.1
        self.pheromone_deposit_cooldown = {}  # Cooldown for pheromone deposits
        self.carrying_food_steps = {}  # Track how long an ant has been carrying food
        self.searching_food_steps = (
            {}
        )  # Track how long an ant has been searching for food
        self.boredom_threshold = 50  # Steps before getting "bored" while carrying food
        self.searching_boredom_threshold = (
            50  # Steps before getting "bored" while searching for food
        )
        self.pheromone_cooldown_value = (
            1  # Cooldown value when resetting pheromone deposit
        )

    def decide_action(self, perception: AntPerception) -> AntAction:
        """Decide the next action for an ant based on current perception"""
        ant_id = perception.ant_id

        # Initialize tracking for new ants
        if ant_id not in self.pheromone_deposit_cooldown:
            self.pheromone_deposit_cooldown[ant_id] = 0
            self.carrying_food_steps[ant_id] = 0
            self.searching_food_steps[ant_id] = 0

        # Track food carrying state changes
        if perception.has_food:
            self.carrying_food_steps[ant_id] += 1
            self.searching_food_steps[ant_id] = (
                0  # Reset searching counter when food is found
            )
        else:
            # Reset counter when not carrying food
            self.carrying_food_steps[ant_id] = 0
            self.searching_food_steps[ant_id] += 1  # Increment searching counter

        # Pick up food if at food cell
        if (
            not perception.has_food
            and (0, 0) in perception.visible_cells
            and perception.visible_cells[(0, 0)] == TerrainType.FOOD
        ):
            self.searching_food_steps[ant_id] = 0
            return AntAction.PICK_UP_FOOD

        # Drop food if at colony cell
        if (
            perception.has_food
            and (0, 0) in perception.visible_cells
            and perception.visible_cells[(0, 0)] == TerrainType.COLONY
        ):
            self.carrying_food_steps[ant_id] = 0  # Reset food carrying counter
            return AntAction.DROP_FOOD

        # Look for food
        if not perception.has_food and perception.can_see_food():
            food_dir = perception.get_food_direction()
            if food_dir is not None:
                # Check if it's time to deposit a pheromone while navigating to visible food
                if self.pheromone_deposit_cooldown[ant_id] <= 0:
                    self.pheromone_deposit_cooldown[ant_id] = (
                        self.pheromone_cooldown_value
                    )  # Reset cooldown
                    return AntAction.DEPOSIT_HOME_PHEROMONE

                # Otherwise, continue moving toward food and decrement cooldown
                self.pheromone_deposit_cooldown[ant_id] -= 1
                current_dir = perception.direction.value
                if food_dir == current_dir:
                    return AntAction.MOVE_FORWARD
                elif (food_dir - current_dir) % 8 <= 4:
                    return AntAction.TURN_RIGHT
                else:  # Turn left if it's shorter
                    return AntAction.TURN_LEFT

        # Look for colony when carrying food
        if perception.has_food and perception.can_see_colony():
            colony_dir = perception.get_colony_direction()
            if colony_dir is not None:
                current_dir = perception.direction.value
                if colony_dir == current_dir:
                    self.pheromone_deposit_cooldown[ant_id] -= 1
                    return AntAction.MOVE_FORWARD
                elif (colony_dir - current_dir) % 8 <= 4:  # Turn right if it's shorter
                    return AntAction.TURN_RIGHT
                else:  # Turn left if it's shorter
                    return AntAction.TURN_LEFT

        # "Boredom" when carrying food too long without finding colony
        if (
            perception.has_food
            and self.carrying_food_steps[ant_id] > self.boredom_threshold
        ):
            self.carrying_food_steps[ant_id] = 0
            action = self._random_movement(perception, forward_chance=0.4)
            return action

        # "Boredom" when searching for food too long
        if (
            not perception.has_food
            and self.searching_food_steps[ant_id] > self.searching_boredom_threshold
        ):
            self.searching_food_steps[ant_id] = 0
            action = self._random_movement(perception, forward_chance=0.4)
            return action

        # Follow food pheromones when not carrying food
        if not perception.has_food and perception.food_pheromone:
            pheromone_dir = self._find_strongest_pheromone(perception.food_pheromone)
            if pheromone_dir is not None:
                # Check if it's time to deposit a pheromone instead of moving
                if self.pheromone_deposit_cooldown[ant_id] <= 0:
                    self.pheromone_deposit_cooldown[ant_id] = (
                        self.pheromone_cooldown_value
                    )  # Reset cooldown
                    return AntAction.DEPOSIT_HOME_PHEROMONE

                # Otherwise, continue following the pheromone and decrement the cooldown
                self.pheromone_deposit_cooldown[ant_id] -= 1
                current_dir = perception.direction.value
                if pheromone_dir == current_dir:
                    return (
                        AntAction.MOVE_FORWARD
                        if self._can_move_forward(perception)
                        else self._random_movement(perception, forward_chance=0.5)
                    )
                elif (
                    pheromone_dir - current_dir
                ) % 8 <= 4:  # Turn right if it's shorter
                    return (
                        AntAction.TURN_RIGHT
                        if self._can_turn_right(perception)
                        else self._random_movement(perception, forward_chance=0.5)
                    )
                else:  # Turn left if it's shorter
                    return (
                        AntAction.TURN_LEFT
                        if self._can_turn_left(perception)
                        else self._random_movement(perception, forward_chance=0.5)
                    )

        # Follow home pheromones when carrying food
        if perception.has_food and perception.home_pheromone:
            pheromone_dir = self._find_strongest_pheromone(perception.home_pheromone)
            if pheromone_dir is not None:
                # Check if it's time to deposit a pheromone instead of moving
                if self.pheromone_deposit_cooldown[ant_id] <= 0:
                    self.pheromone_deposit_cooldown[ant_id] = (
                        self.pheromone_cooldown_value
                    )  # Reset cooldown
                    return AntAction.DEPOSIT_FOOD_PHEROMONE

                # Otherwise, continue following the pheromone and decrement the cooldown
                self.pheromone_deposit_cooldown[ant_id] -= 1
                current_dir = perception.direction.value

                # Determine movement based on pheromone direction
                if pheromone_dir == current_dir:
                    return (
                        AntAction.MOVE_FORWARD
                        if self._can_move_forward(perception)
                        else self._random_movement(perception, forward_chance=0.5)
                    )
                elif (
                    pheromone_dir - current_dir
                ) % 8 <= 4:  # Turn right if it's shorter
                    return (
                        AntAction.TURN_RIGHT
                        if self._can_turn_right(perception)
                        else self._random_movement(perception, forward_chance=0.5)
                    )
                else:  # Turn left if it's shorter
                    return (
                        AntAction.TURN_LEFT
                        if self._can_turn_left(perception)
                        else self._random_movement(perception, forward_chance=0.5)
                    )

        # Deposit pheromones with cooldown
        if self.pheromone_deposit_cooldown[ant_id] <= 0:
            self.pheromone_deposit_cooldown[ant_id] = (
                self.pheromone_cooldown_value
            )  # Reset cooldown
            if perception.has_food:
                return AntAction.DEPOSIT_FOOD_PHEROMONE
            else:
                return AntAction.DEPOSIT_HOME_PHEROMONE
        else:
            self.pheromone_deposit_cooldown[ant_id] -= 1

        # Random movement as fallback
        action = self._random_movement(perception, forward_chance=0.7)
        return action

    def _random_movement(self, perception, forward_chance: float = 0.5):
        """Random movement with forward bias, avoiding walls"""

        can_move_forward = self._can_move_forward(perception)
        can_turn_left = self._can_turn_left(perception)
        can_turn_right = self._can_turn_right(perception)

        available_actions = []
        if can_move_forward:
            available_actions.append(AntAction.MOVE_FORWARD)
        if can_turn_left:
            available_actions.append(AntAction.TURN_LEFT)
        if can_turn_right:
            available_actions.append(AntAction.TURN_RIGHT)

        # If no valid actions are available, return NO_ACTION
        if not available_actions:
            return AntAction.TURN_LEFT

        # Bias toward moving forward if possible

        if (
            can_move_forward and random.random() < forward_chance
        ):  # Higher probability to move forward
            return AntAction.MOVE_FORWARD
        else:
            # Select from remaining valid actions
            remaining_actions = []
            if can_turn_left:
                remaining_actions.append(AntAction.TURN_LEFT)
            if can_turn_right:
                remaining_actions.append(AntAction.TURN_RIGHT)

            if remaining_actions:
                return random.choice(remaining_actions)
            elif can_move_forward:  # If can't turn but can move forward
                return AntAction.MOVE_FORWARD
            else:
                return AntAction.TURN_LEFT  # No valid moves

    def _find_strongest_pheromone(self, pheromone_map):
        """Find the strongest pheromone direction"""
        if not pheromone_map:
            return None

        max_strength = self.pheromone_threshold
        strongest_pos = None

        # Find the strongest pheromone signal using the visible cells
        for pos, strength in pheromone_map.items():
            # Limit perception range for more direct movement
            if strength > max_strength:
                max_strength = strength
                strongest_pos = pos

        if strongest_pos:
            return self._calculate_direction(strongest_pos[0], strongest_pos[1])
        return None

    def _calculate_direction(self, dx, dy):
        """Calculate a direction value based on dx, dy coordinates"""
        if dx == 0 and dy < 0:
            return Direction.NORTH.value
        elif dx > 0 and dy < 0:
            return Direction.NORTHEAST.value
        elif dx > 0 and dy == 0:
            return Direction.EAST.value
        elif dx > 0 and dy > 0:
            return Direction.SOUTHEAST.value
        elif dx == 0 and dy > 0:
            return Direction.SOUTH.value
        elif dx < 0 and dy > 0:
            return Direction.SOUTHWEST.value
        elif dx < 0 and dy == 0:
            return Direction.WEST.value
        elif dx < 0 and dy < 0:
            return Direction.NORTHWEST.value
        return Direction.NORTH.value

    def _can_move_forward(self, perception):
        forward_delta = Direction.get_delta(perception.direction)
        return (
            forward_delta in perception.visible_cells
            and perception.visible_cells[forward_delta] != TerrainType.WALL
        )

    def _can_turn_left(self, perception):
        left_dir = Direction.get_left(perception.direction)
        left_delta = Direction.get_delta(left_dir)
        return (
            left_delta in perception.visible_cells
            and perception.visible_cells[left_delta] != TerrainType.WALL
        )

    def _can_turn_right(self, perception):
        right_dir = Direction.get_right(perception.direction)
        right_delta = Direction.get_delta(right_dir)
        return (
            right_delta in perception.visible_cells
            and perception.visible_cells[right_delta] != TerrainType.WALL
        )