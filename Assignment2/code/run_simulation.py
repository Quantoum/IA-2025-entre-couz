#!/usr/bin/env python3
"""
Simulation script for running multiple games between agents and collecting statistics.
This script runs without graphics for maximum speed.
"""

import time
import argparse
import multiprocessing
import numpy as np
from copy import deepcopy
from agent import Agent
from random_agent import RandomAgent
from game_manager import TextGameManager
import matplotlib.pyplot as plt
from tqdm import tqdm  # Optional dependency for progress bars

class SimulationManager:
    """
    Manages running multiple games between agents and collecting statistics.
    """
    
    def __init__(self, num_games=100, time_limit=300, verbose=False):
        self.num_games = num_games
        self.time_limit = time_limit
        self.verbose = verbose
        self.results = []
        self.game_lengths = []
        self.strategy_usage = {}
        self.wins_by_strategy = {'red': {}, 'black': {}}
        self.total_time = 0
        
    def run_single_game(self, game_id, red_agent, black_agent):
        """Run a single game and return results."""
        game_start = time.time()
        
        # Create agents
        agent_1 = red_agent(player=1)
        agent_2 = black_agent(player=-1)
        
        # Run game without display
        game_manager = TextGameManager(agent_1, agent_2, time_limit=self.time_limit, display=False)
        score_1, score_2 = game_manager.play()
        
        game_time = time.time() - game_start
        
        # Collect statistics from agents if they support it
        strategies_used = {}
        if hasattr(agent_1, 'get_performance_summary'):
            red_stats = agent_1.get_performance_summary()
            strategies_used['red'] = red_stats.get('strategy_usage', {})
        
        if hasattr(agent_2, 'get_performance_summary'):
            black_stats = agent_2.get_performance_summary()
            strategies_used['black'] = black_stats.get('strategy_usage', {})
        
        # Determine winner
        if score_1 > score_2:
            winner = 'red'
        elif score_2 > score_1:
            winner = 'black'
        else:
            winner = 'draw'
            
        return {
            'game_id': game_id,
            'score_red': score_1,
            'score_black': score_2,
            'winner': winner,
            'game_time': game_time,
            'game_length': getattr(game_manager.state, 'turn', 0),
            'strategies_used': strategies_used
        }
    
    def run_simulation(self, red_agent_class=Agent, black_agent_class=RandomAgent, parallel=False, num_workers=None):
        """Run multiple games and collect statistics."""
        start_time = time.time()
        self.results = []
        
        if parallel and num_workers:
            # Parallel execution using multiprocessing
            with multiprocessing.Pool(processes=num_workers) as pool:
                args = [(i, red_agent_class, black_agent_class) for i in range(self.num_games)]
                self.results = list(tqdm(pool.starmap(self.run_single_game, args), total=self.num_games))
        else:
            # Sequential execution
            for i in tqdm(range(self.num_games)) if tqdm else range(self.num_games):
                result = self.run_single_game(i, red_agent_class, black_agent_class)
                self.results.append(result)
                
                if self.verbose:
                    print(f"Game {i+1}: Red {result['score_red']} - Black {result['score_black']} | Winner: {result['winner']} | Time: {result['game_time']:.2f}s")
        
        self.total_time = time.time() - start_time
        self._process_results()
        
        return self.get_statistics()
    
    def _process_results(self):
        """Process the raw results into useful statistics."""
        # Extract game lengths
        self.game_lengths = [r['game_length'] for r in self.results]
        
        # Count strategy usage
        self.strategy_usage = {'red': {}, 'black': {}}
        self.wins_by_strategy = {'red': {}, 'black': {}}
        
        for r in self.results:
            strategies = r.get('strategies_used', {})
            winner = r['winner']
            
            # Process red strategies
            if 'red' in strategies:
                for strategy, count in strategies['red'].items():
                    self.strategy_usage['red'][strategy] = self.strategy_usage['red'].get(strategy, 0) + count
                    
                    # Track wins by strategy
                    if winner == 'red':
                        self.wins_by_strategy['red'][strategy] = self.wins_by_strategy['red'].get(strategy, 0) + count
            
            # Process black strategies
            if 'black' in strategies:
                for strategy, count in strategies['black'].items():
                    self.strategy_usage['black'][strategy] = self.strategy_usage['black'].get(strategy, 0) + count
                    
                    # Track wins by strategy
                    if winner == 'black':
                        self.wins_by_strategy['black'][strategy] = self.wins_by_strategy['black'].get(strategy, 0) + count
    
    def get_statistics(self):
        """Get comprehensive statistics about the simulation results."""
        if not self.results:
            return None
            
        # Count wins
        red_wins = sum(1 for r in self.results if r['winner'] == 'red')
        black_wins = sum(1 for r in self.results if r['winner'] == 'black')
        draws = sum(1 for r in self.results if r['winner'] == 'draw')
        
        # Calculate win percentages
        red_win_pct = red_wins / self.num_games * 100
        black_win_pct = black_wins / self.num_games * 100
        draw_pct = draws / self.num_games * 100
        
        # Calculate game times
        game_times = [r['game_time'] for r in self.results]
        avg_game_time = np.mean(game_times) if game_times else 0
        min_game_time = min(game_times) if game_times else 0
        max_game_time = max(game_times) if game_times else 0
        
        # Calculate game lengths
        avg_game_length = np.mean(self.game_lengths) if self.game_lengths else 0
        min_game_length = min(self.game_lengths) if self.game_lengths else 0
        max_game_length = max(self.game_lengths) if self.game_lengths else 0
        
        # Calculate strategy effectiveness (win rate per strategy)
        strategy_win_rate = {'red': {}, 'black': {}}
        for color in ['red', 'black']:
            for strategy, count in self.strategy_usage.get(color, {}).items():
                if count > 0:
                    win_count = self.wins_by_strategy.get(color, {}).get(strategy, 0)
                    strategy_win_rate[color][strategy] = (win_count / count) * 100
        
        return {
            'total_games': self.num_games,
            'red_wins': red_wins,
            'black_wins': black_wins,
            'draws': draws,
            'red_win_pct': red_win_pct,
            'black_win_pct': black_win_pct,
            'draw_pct': draw_pct,
            'game_times': {
                'avg': avg_game_time,
                'min': min_game_time,
                'max': max_game_time,
                'total': self.total_time
            },
            'game_lengths': {
                'avg': avg_game_length,
                'min': min_game_length,
                'max': max_game_length
            },
            'strategy_usage': self.strategy_usage,
            'wins_by_strategy': self.wins_by_strategy,
            'strategy_win_rate': strategy_win_rate
        }
    
    def print_statistics(self, stats=None):
        """Print formatted statistics to console."""
        if stats is None:
            stats = self.get_statistics()
            
        if not stats:
            print("No statistics available. Run simulation first.")
            return
            
        print("\n===== SIMULATION RESULTS =====")
        print(f"Total Games: {stats['total_games']}")
        print(f"Red Wins: {stats['red_wins']} ({stats['red_win_pct']:.2f}%)")
        print(f"Black Wins: {stats['black_wins']} ({stats['black_win_pct']:.2f}%)")
        print(f"Draws: {stats['draws']} ({stats['draw_pct']:.2f}%)")
        
        print("\n----- Game Time Statistics -----")
        print(f"Average Game Time: {stats['game_times']['avg']:.2f} seconds")
        print(f"Minimum Game Time: {stats['game_times']['min']:.2f} seconds")
        print(f"Maximum Game Time: {stats['game_times']['max']:.2f} seconds")
        print(f"Total Simulation Time: {stats['game_times']['total']:.2f} seconds")
        
        print("\n----- Game Length Statistics -----")
        print(f"Average Game Length: {stats['game_lengths']['avg']:.2f} turns")
        print(f"Minimum Game Length: {stats['game_lengths']['min']} turns")
        print(f"Maximum Game Length: {stats['game_lengths']['max']} turns")
        
        # Print strategy usage for red player
        if stats['strategy_usage'].get('red'):
            print("\n----- Red Player Strategy Usage -----")
            total_red = sum(stats['strategy_usage']['red'].values())
            for strategy, count in stats['strategy_usage']['red'].items():
                usage_pct = (count / total_red) * 100
                win_rate = stats['strategy_win_rate']['red'].get(strategy, 0)
                print(f"{strategy}: {count} uses ({usage_pct:.2f}%) - Win Rate: {win_rate:.2f}%")
                
        # Print strategy usage for black player
        if stats['strategy_usage'].get('black'):
            print("\n----- Black Player Strategy Usage -----")
            total_black = sum(stats['strategy_usage']['black'].values())
            for strategy, count in stats['strategy_usage']['black'].items():
                usage_pct = (count / total_black) * 100
                win_rate = stats['strategy_win_rate']['black'].get(strategy, 0)
                print(f"{strategy}: {count} uses ({usage_pct:.2f}%) - Win Rate: {win_rate:.2f}%")
    
    def plot_results(self, stats=None):
        """Generate plots visualizing the simulation results."""
        try:
            import matplotlib.pyplot as plt
            
            if stats is None:
                stats = self.get_statistics()
                
            if not stats:
                print("No statistics available. Run simulation first.")
                return
                
            # Plot 1: Win distribution pie chart
            plt.figure(figsize=(10, 8))
            plt.subplot(2, 2, 1)
            labels = ['Red Wins', 'Black Wins', 'Draws']
            sizes = [stats['red_win_pct'], stats['black_win_pct'], stats['draw_pct']]
            colors = ['red', 'black', 'gray']
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.title('Game Outcome Distribution')
            
            # Plot 2: Game lengths histogram
            plt.subplot(2, 2, 2)
            plt.hist(self.game_lengths, bins=20, color='blue', alpha=0.7)
            plt.axvline(stats['game_lengths']['avg'], color='red', linestyle='dashed', linewidth=1)
            plt.xlabel('Game Length (turns)')
            plt.ylabel('Frequency')
            plt.title('Game Length Distribution')
            
            # Plot 3: Red player strategy usage
            if stats['strategy_usage'].get('red'):
                plt.subplot(2, 2, 3)
                strategies = list(stats['strategy_usage']['red'].keys())
                counts = list(stats['strategy_usage']['red'].values())
                plt.bar(strategies, counts, color='red', alpha=0.7)
                plt.xlabel('Strategy')
                plt.ylabel('Usage Count')
                plt.title('Red Player Strategy Usage')
                plt.xticks(rotation=45, ha='right')
            
            # Plot 4: Black player strategy usage
            if stats['strategy_usage'].get('black'):
                plt.subplot(2, 2, 4)
                strategies = list(stats['strategy_usage']['black'].keys())
                counts = list(stats['strategy_usage']['black'].values())
                plt.bar(strategies, counts, color='gray', alpha=0.7)
                plt.xlabel('Strategy')
                plt.ylabel('Usage Count')
                plt.title('Black Player Strategy Usage')
                plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig('simulation_results.png')
            plt.close()
            print("Plot saved as 'simulation_results.png'")
        except ImportError:
            print("Matplotlib not available. Install it to plot results.")

def main():
    parser = argparse.ArgumentParser(description='Run a simulation of multiple games between agents')
    parser.add_argument('--num-games', type=int, default=100, help='Number of games to play')
    parser.add_argument('--time-limit', type=int, default=300, help='Time limit per player per game (seconds)')
    parser.add_argument('--verbose', action='store_true', help='Print detailed game results')
    parser.add_argument('--parallel', action='store_true', help='Run games in parallel')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes for parallel execution')
    parser.add_argument('--no-plot', action='store_true', help='Do not generate plots')
    
    args = parser.parse_args()
    
    # Set default workers to CPU count if parallel is enabled
    if args.parallel and args.workers is None:
        args.workers = multiprocessing.cpu_count()
    
    # Initialize the simulation manager
    sim_manager = SimulationManager(
        num_games=args.num_games,
        time_limit=args.time_limit,
        verbose=args.verbose
    )
    
    print(f"Starting simulation of {args.num_games} games...")
    print(f"Agent 1 (Red): Hybrid Agent")
    print(f"Agent 2 (Black): Random Agent")
    print(f"Time limit per player: {args.time_limit} seconds")
    print(f"Execution mode: {'Parallel (' + str(args.workers) + ' workers)' if args.parallel else 'Sequential'}")
    
    # Run the simulation
    stats = sim_manager.run_simulation(
        red_agent_class=Agent,
        black_agent_class=RandomAgent,
        parallel=args.parallel,
        num_workers=args.workers
    )
    
    # Print statistics
    sim_manager.print_statistics(stats)
    
    # Plot results if requested
    if not args.no_plot:
        sim_manager.plot_results(stats)
    
    return stats

if __name__ == "__main__":
    main() 