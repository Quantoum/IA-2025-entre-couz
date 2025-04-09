#
# LINFO 1361 - Artificial Intelligence 
# Fenix game - April 2025
# Author: Arnaud Ullens, Quentin de Pierpont
# 

import time
from random import choice
from mcts import MonteCarloTreeSearchNode
from alpha_beta import AlphaBeta
import warnings
from consts import MAX_MCTS_ITERATIONS
from fenix import FenixAction
class Agent:
    def __init__(self, player):
        self.player = player
        self.number_of_registered_moves = 2 # entre 0 et 5
        
class HybridAgent(Agent):
    """
    A hybrid agent that combines MCTS and Alpha-Beta pruning based on the game phase.
    """
    def __init__(self, player):
        super().__init__(player)
        self.transposition_table = {}  # Shared transposition table for both strategies
        # Thresholds for complexity and phase detection
        self.EARLY_GAME_THRESHOLD = 10  # turn < 10 is early game
        self.LATE_GAME_THRESHOLD = 30   # turn >= 30 is late game
        self.HIGH_COMPLEXITY_THRESHOLD = 15  # > 15 legal moves means high complexity
        # Keep track of time usage
        self.last_execution_time = 0
        
    def determine_phase(self, state):
        """
        Determine the current game phase based on turn number and piece count.
        
        Returns:
            str: 'early', 'mid', or 'late'
        """
        # Early game is determined by turn number
        if state.turn < self.EARLY_GAME_THRESHOLD:
            return 'early'
            
        # Count the pieces on the board
        piece_count = len(state.pieces)
        
        # Late game is determined by turn number or low piece count
        if state.turn >= self.LATE_GAME_THRESHOLD or piece_count <= 10:
            return 'late'
            
        # Otherwise, it's mid-game
        return 'mid'
    
    def evaluate_complexity(self, state):
        """
        Evaluate the complexity of the current state based on the number of legal moves.
        
        Returns:
            str: 'high' or 'low'
        """
        legal_moves = state.actions()
        return 'high' if len(legal_moves) > self.HIGH_COMPLEXITY_THRESHOLD else 'low'
        
    def select_strategy(self, state, remaining_time):
        """
        Select which strategy to use (MCTS or Alpha-Beta) based on game phase, 
        complexity, and remaining time.
        
        Returns:
            str: 'mcts' or 'alpha_beta'
        """
        # Critical time scenario - use fast approach
        if remaining_time <= 1:
            return 'random'
        elif remaining_time <= 10:
            return 'alpha_beta_fast'
            
        # Determine phase and complexity
        phase = self.determine_phase(state)
        complexity = self.evaluate_complexity(state)
        
        # Use registered moves at the start if applicable
        if phase == 'early' and self.number_of_registered_moves*2 > state.turn:
            return 'opening'
            
        # Strategy selection logic
        if phase == 'early':
            # Early game: prefer Alpha-Beta for tactical precision
            return 'alpha_beta'
        elif phase == 'late':
            # Late game: prefer Alpha-Beta for endgame precision
            return 'alpha_beta'
        else:  # mid-game
            # Mid-game: use MCTS for high complexity, Alpha-Beta for low
            return 'mcts' if complexity == 'high' else 'alpha_beta'
            
    def create_enhanced_mcts(self, state):
        """
        Create an enhanced MCTS node with a custom rollout policy using Alpha-Beta heuristics.
        """
        # Create a custom MonteCarloTreeSearchNode that uses Alpha-Beta heuristics
        class EnhancedMCTSNode(MonteCarloTreeSearchNode):
            def __init__(self, state, player, alpha_beta_agent, parent=None, parent_action=None, max_iterations=MAX_MCTS_ITERATIONS):
                super().__init__(state, player, parent, parent_action, max_iterations)
                self.alpha_beta_agent = alpha_beta_agent
                
            def rollout_policy(self, possible_moves):
                """
                Enhanced rollout policy that uses Alpha-Beta heuristics to guide the selection.
                """
                # If there are too many moves or time is critical, use random selection
                if len(possible_moves) > 30:
                    return choice(possible_moves)
                    
                # Evaluate moves with heuristics and select proportional to their value
                move_values = []
                for move in possible_moves:
                    next_state = self.state.result(move)
                    # Use material and positional heuristics
                    material_value = self.alpha_beta_agent.materialHeuristic(next_state)
                    positional_value = self.alpha_beta_agent.positionalHeuristic(next_state)
                    move_values.append(3 * material_value + positional_value)
                    
                # Find the best moves (top 3 if available)
                if not move_values:
                    return choice(possible_moves)
                
                # Get the top 3 or fewer moves
                num_top_moves = min(3, len(possible_moves))
                
                # Sort moves by their values and get the top ones
                sorted_moves = [move for _, move in sorted(zip(move_values, possible_moves), key=lambda x: x[0], reverse=True)]
                top_moves = sorted_moves[:num_top_moves]
                
                # Choose from top moves with preference to the best
                return choice(top_moves)
        
        # Create an Alpha-Beta agent for heuristics
        alpha_beta = AlphaBeta(self.player)
        
        # Return the enhanced MCTS node
        return EnhancedMCTSNode(state=state, player=self.player, alpha_beta_agent=alpha_beta, max_iterations=MAX_MCTS_ITERATIONS)

    def act(self, state, remaining_time):
        """
        Choose an action using either MCTS or Alpha-Beta based on game phase and complexity.
        """
        start_time = time.time()
        
        # Make sure we have legal moves
        legal_moves = state.actions()
        if not legal_moves:
            warnings.warn("No legal moves available!")
            return None
            
        # Select strategy based on game state
        strategy = self.select_strategy(state, remaining_time)
        
        action = None
        try:
            if strategy == 'random':
                # Critical time - use random move
                print("Critical time - selecting random move.")
                action = choice(legal_moves)
            elif strategy == 'opening':
                # Use predetermined opening move
                action = self.start_game(state)
            elif strategy == 'alpha_beta_fast':
                # Use Alpha-Beta with limited depth for fast decisions
                print("Low time - using Alpha-Beta with limited depth.")
                alpha_beta = AlphaBeta(self.player, max_depth=4)
                action = alpha_beta.best_action(state)
            elif strategy == 'alpha_beta':
                # Use full Alpha-Beta search
                try:
                    print("Using Alpha-Beta search...")
                    max_depth = 7 if remaining_time > 30 else 5
                    alpha_beta = AlphaBeta(self.player, max_depth=max_depth)
                    action = alpha_beta.best_action(state)
                except Exception as e:
                    print(f"Error during Alpha-Beta: {e}")
                    # Fall back to simpler Alpha-Beta
                    alpha_beta = AlphaBeta(self.player, max_depth=3)
                    action = alpha_beta.best_action(state)
            elif strategy == 'mcts':
                # Use enhanced MCTS
                try:
                    print("Using Enhanced MCTS...")
                    mcts_root = self.create_enhanced_mcts(state)
                    selected_node = mcts_root.best_action()
                    action = selected_node.parent_action
                except RecursionError as e:
                    warnings.warn(f"Recursion error during MCTS: {e}. Falling back to Alpha-Beta.")
                    alpha_beta = AlphaBeta(self.player, max_depth=4)
                    action = alpha_beta.best_action(state)
                except Exception as e:
                    print(f"Error during MCTS: {e}")
                    alpha_beta = AlphaBeta(self.player, max_depth=3)
                    action = alpha_beta.best_action(state)
        except Exception as e:
            # Ultimate fallback - random move
            print(f"Unexpected error in strategy execution: {e}")
            action = choice(legal_moves)
        
        # Track execution time
        self.last_execution_time = time.time() - start_time
        print(f"Move decision took {self.last_execution_time:.2f} seconds, strategy: {strategy}")
        
        return action
    
    def start_game(self, state):
        """
        Return a predefined opening move from a registered sequence.
        If the move is not valid for the current state, fall back to Alpha-Beta.
        
        Returns:
            FenixAction: The selected opening move
        """
        # First set of opening moves (commented out for reference)
        """
        registered_moves = {
            0: ((0,0),(1,0)), # red general
            1: ((6,7),(5,7)), # black general 
            2: ((0,1),(1,1)), # red general
            3: ((6,6),(5,6)), # black general
            4: ((0,2),(1,2)), # red general
            5: ((6,5),(5,5)), # black general
            6: ((0,3),(0,4)), # red general
            7: ((6,4),(6,3)), # black general
            8: ((2,0),(1,0)), # red king
            9: ((4,7),(5,7))  # black king
        }
        """
        # Current set of opening moves
        registered_moves = {
            0: ((1,0),(0,0)), # red general
            1: ((5,7),(6,7)), # black general 
            2: ((0,1),(0,0)), # red king
            3: ((6,6),(6,7)), # black king
            4: ((1,1),(2,1)), # red general
            5: ((5,6),(4,6)), # black general
            6: ((0,2),(0,3)), # red general
            7: ((5,5),(4,5)), # black general
            8: ((1,2),(2,2)), # red general
            9: ((6,5),(6,4))  # black general
        }
        
        # Try to get the registered move for the current turn
        move_data = registered_moves.get(state.turn)
        
        if move_data:
            start, end = move_data
            # Create the FenixAction
            action = FenixAction(start, end, removed=frozenset())
            
            # Verify if the move is valid in the current state
            legal_moves = state.actions()
            if action in legal_moves:
                print(f"Using registered opening move: {action}")
                return action
                
            # If not valid, warn and fall back to Alpha-Beta
            print(f"Registered move {action} not valid in current state. Falling back to Alpha-Beta.")
        else:
            print(f"No registered move for turn {state.turn}. Falling back to Alpha-Beta.")
            
        # Fall back to Alpha-Beta with low depth for quick decision
        alpha_beta = AlphaBeta(self.player, max_depth=4)
        return alpha_beta.best_action(state)
