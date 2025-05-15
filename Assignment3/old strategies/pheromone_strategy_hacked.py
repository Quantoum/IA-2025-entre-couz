from datetime import datetime
import logging
import os
import random

from ant import AntStrategy
from common import AntAction, AntPerception, Direction, TerrainType


# Configure logging
def setup_logging():
    """Set up logging to both console and file"""
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create a unique log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"ant_simulation_{timestamp}.log")

    # Configure root logger to handle both file and console output
    logger = logging.getLogger("ant_colony")
    logger.setLevel(logging.INFO)

    # Clear any existing handlers (to avoid duplicate logging)
    if logger.handlers:
        logger.handlers.clear()

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    # logger.addHandler(console_handler)

    # Log startup message
    logger.info(f"Ant simulation log started. Logging to: {log_file}")

    return logger, timestamp


# Initialize the main logger and get timestamp
logger, timestamp = setup_logging()

# Dictionary to store per-ant loggers
ant_loggers = {}


def get_ant_logger(ant_id):
    """Get or create a logger for a specific ant"""
    if ant_id in ant_loggers:
        return ant_loggers[ant_id]

    # Create logs directory if it doesn't exist
    log_dir = "logs/ants"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create a logger for this ant
    ant_logger = logging.getLogger(f"ant_{ant_id}")
    ant_logger.setLevel(logging.INFO)

    # Clear any existing handlers
    if ant_logger.handlers:
        ant_logger.handlers.clear()

    # Create file handler for this ant
    ant_log_file = os.path.join(log_dir, f"ant_{ant_id}_{timestamp}.log")
    file_handler = logging.FileHandler(ant_log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Add handler to logger
    ant_logger.addHandler(file_handler)

    # Store and return the logger
    ant_loggers[ant_id] = ant_logger
    return ant_logger


class PheromoneStrategy(AntStrategy):
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

        # Get logger for this ant
        ant_log = get_ant_logger(ant_id)

        # Initialize tracking for new ants
        if ant_id not in self.pheromone_deposit_cooldown:
            self.pheromone_deposit_cooldown[ant_id] = 0
            self.carrying_food_steps[ant_id] = 0
            self.searching_food_steps[ant_id] = 0
            logger.info(
                f"Ant #{ant_id}: Initialized with direction {perception.direction}"
            )
            ant_log.info(f"Initialized with direction {perception.direction}")

        # Debug print visible cells to diagnose terrain perception
        ant_log.debug(f"Visible cells: {perception.visible_cells}")

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
            logger.info(f"Ant #{ant_id}: Standing on food, picking up")
            ant_log.info("Standing on food, picking up")
            return AntAction.PICK_UP_FOOD

        # Drop food if at colony cell
        if (
            perception.has_food
            and (0, 0) in perception.visible_cells
            and perception.visible_cells[(0, 0)] == TerrainType.COLONY
        ):
            logger.info(f"Ant #{ant_id}: Standing at colony, dropping food")
            ant_log.info("Standing at colony, dropping food")
            self.carrying_food_steps[ant_id] = 0  # Reset food carrying counter
            return AntAction.DROP_FOOD

        # Look for food
        if not perception.has_food and perception.can_see_food():
            logger.info(f"Ant #{ant_id}: Can see food, getting direction")
            ant_log.info("Can see food, getting direction")
            food_dir = perception.get_food_direction()
            if food_dir is not None:
                # Check if it's time to deposit a pheromone while navigating to visible food
                if self.pheromone_deposit_cooldown[ant_id] <= 0:
                    self.pheromone_deposit_cooldown[ant_id] = (
                        self.pheromone_cooldown_value
                    )  # Reset cooldown
                    logger.info(
                        f"Ant #{ant_id}: Depositing home pheromone while moving toward visible food"
                    )
                    ant_log.info(
                        "Depositing home pheromone while moving toward visible food"
                    )
                    return AntAction.DEPOSIT_HOME_PHEROMONE

                # Otherwise, continue moving toward food and decrement cooldown
                self.pheromone_deposit_cooldown[ant_id] -= 1
                current_dir = perception.direction.value
                if food_dir == current_dir:
                    logger.info(f"Ant #{ant_id}: Food in front, moving forward")
                    ant_log.info("Food in front, moving forward")
                    return AntAction.MOVE_FORWARD
                elif (food_dir - current_dir) % 8 <= 4:  # Turn right if it's shorter
                    logger.info(f"Ant #{ant_id}: Food to the right, turning right")
                    ant_log.info("Food to the right, turning right")
                    return AntAction.TURN_RIGHT
                else:  # Turn left if it's shorter
                    logger.info(f"Ant #{ant_id}: Food to the left, turning left")
                    ant_log.info("Food to the left, turning left")
                    return AntAction.TURN_LEFT

        # Look for colony when carrying food
        if perception.has_food and perception.can_see_colony():
            logger.info(f"Ant #{ant_id}: Can see colony, returning with food")
            ant_log.info("Can see colony, returning with food")
            colony_dir = perception.get_colony_direction()
            if colony_dir is not None:
                current_dir = perception.direction.value
                if colony_dir == current_dir:
                    self.pheromone_deposit_cooldown[ant_id] -= 1
                    logger.info(f"Ant #{ant_id}: Colony in front, moving forward")
                    ant_log.info("Colony in front, moving forward")
                    return AntAction.MOVE_FORWARD
                elif (colony_dir - current_dir) % 8 <= 4:  # Turn right if it's shorter
                    logger.info(f"Ant #{ant_id}: Colony to the right, turning right")
                    ant_log.info("Colony to the right, turning right")
                    return AntAction.TURN_RIGHT
                else:  # Turn left if it's shorter
                    logger.info(f"Ant #{ant_id}: Colony to the left, turning left")
                    ant_log.info("Colony to the left, turning left")
                    return AntAction.TURN_LEFT

        # "Boredom" when carrying food too long without finding colony
        if (
            perception.has_food
            and self.carrying_food_steps[ant_id] > self.boredom_threshold
        ):

            logger.info(
                f"Ant #{ant_id}: Bored after carrying food for {self.carrying_food_steps[ant_id]} steps, making random move"
            )
            ant_log.info(
                f"Bored after carrying food for {self.carrying_food_steps[ant_id]} steps, making random move"
            )
            self.carrying_food_steps[ant_id] = 0
            action = self._random_movement(perception, forward_chance=0.4)
            logger.info(f"Ant #{ant_id}: Made random move: {action}")
            ant_log.info(f"Made random move: {action}")
            return action

        # "Boredom" when searching for food too long
        if (
            not perception.has_food
            and self.searching_food_steps[ant_id] > self.searching_boredom_threshold
        ):
            # Every 15 steps after threshold, make a random move
            logger.info(
                f"Ant #{ant_id}: Bored after searching for food for {self.searching_food_steps[ant_id]} steps, making random move"
            )
            ant_log.info(
                f"Bored after searching for food for {self.searching_food_steps[ant_id]} steps, making random move"
            )
            self.searching_food_steps[ant_id] = 0
            action = self._random_movement(perception, forward_chance=0.4)
            logger.info(f"Ant #{ant_id}: Made random move: {action}")
            ant_log.info(f"Made random move: {action}")
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
                    logger.info(
                        f"Ant #{ant_id}: Depositing home pheromone while following food trail"
                    )
                    ant_log.info("Depositing home pheromone while following food trail")
                    return AntAction.DEPOSIT_HOME_PHEROMONE

                # Otherwise, continue following the pheromone and decrement the cooldown
                self.pheromone_deposit_cooldown[ant_id] -= 1
                current_dir = perception.direction.value
                if pheromone_dir == current_dir:
                    logger.info(
                        f"Ant #{ant_id}: Food pheromone in front, moving forward"
                    )
                    ant_log.info("Food pheromone in front, moving forward")
                    return (
                        AntAction.MOVE_FORWARD
                        if self._can_move_forward(perception)
                        else self._random_movement(perception, forward_chance=0.5)
                    )
                elif (
                    pheromone_dir - current_dir
                ) % 8 <= 4:  # Turn right if it's shorter
                    logger.info(
                        f"Ant #{ant_id}: Food pheromone to the right, turning right"
                    )
                    ant_log.info("Food pheromone to the right, turning right")
                    return (
                        AntAction.TURN_RIGHT
                        if self._can_turn_right(perception)
                        else self._random_movement(perception, forward_chance=0.5)
                    )
                else:  # Turn left if it's shorter
                    logger.info(
                        f"Ant #{ant_id}: Food pheromone to the left, turning left"
                    )
                    ant_log.info("Food pheromone to the left, turning left")
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
                    logger.info(
                        f"Ant #{ant_id}: Depositing food pheromone while following home trail"
                    )
                    ant_log.info("Depositing food pheromone while following home trail")
                    return AntAction.DEPOSIT_FOOD_PHEROMONE

                # Otherwise, continue following the pheromone and decrement the cooldown
                self.pheromone_deposit_cooldown[ant_id] -= 1
                current_dir = perception.direction.value

                # Determine movement based on pheromone direction
                if pheromone_dir == current_dir:
                    logger.info(
                        f"Ant #{ant_id}: Home pheromone in front, moving forward"
                    )
                    ant_log.info("Home pheromone in front, moving forward")
                    return (
                        AntAction.MOVE_FORWARD
                        if self._can_move_forward(perception)
                        else self._random_movement(perception, forward_chance=0.5)
                    )
                elif (
                    pheromone_dir - current_dir
                ) % 8 <= 4:  # Turn right if it's shorter
                    logger.info(
                        f"Ant #{ant_id}: Home pheromone to the right, turning right"
                    )
                    ant_log.info("Home pheromone to the right, turning right")
                    return (
                        AntAction.TURN_RIGHT
                        if self._can_turn_right(perception)
                        else self._random_movement(perception, forward_chance=0.5)
                    )
                else:  # Turn left if it's shorter
                    logger.info(
                        f"Ant #{ant_id}: Home pheromone to the left, turning left"
                    )
                    ant_log.info("Home pheromone to the left, turning left")
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
                logger.info(f"Ant #{ant_id}: Depositing food pheromone")
                ant_log.info("Depositing food pheromone")
                return AntAction.DEPOSIT_FOOD_PHEROMONE
            else:
                logger.info(f"Ant #{ant_id}: Depositing home pheromone")
                ant_log.info("Depositing home pheromone")
                return AntAction.DEPOSIT_HOME_PHEROMONE
        else:
            self.pheromone_deposit_cooldown[ant_id] -= 1

        # Random movement as fallback
        action = self._random_movement(perception, forward_chance=0.7)
        ant_log.info(f"Random movement: {action}")
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
            logger.info("No valid actions available")
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