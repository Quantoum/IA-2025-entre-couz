from numpy import random, uint64
from consts import *

_ZOBRIST_INITIALIZED = False
_zobrist_pieces = None
_zobrist_player = None

def _init_global_zobrist_tables():
    """Initialize global Zobrist tables with fixed seed."""
    global _ZOBRIST_INITIALIZED, _zobrist_pieces, _zobrist_player
    
    if not _ZOBRIST_INITIALIZED:
        rng = random.default_rng(seed=42)
        
        _zobrist_pieces = rng.integers(0, 2**63, size=(2, 3, 7, 8), dtype=uint64)
        _zobrist_player = rng.integers(0, 2**63, size=2, dtype=uint64)
        
        _ZOBRIST_INITIALIZED = True

_init_global_zobrist_tables()

class TranspositionTable:
    """
    Transposition table (zobrist hash).
    Each entry contains:
    - hash_key: Zobrist hash of the position
    - value: best value found
    - depth: search depth
    - flag: EXACT, LOWER_BOUND, or UPPER_BOUND
    - best_move: best move found
    """

    EXACT = 0
    LOWER_BOUND = 1
    UPPER_BOUND = 2

    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.table = {}
        self.hits = 0
        self.misses = 0

    def compute_hash(self, state):
        """
        Compute Zobrist hash for a game state.
        Returns the Zobrist hash value
        """
        hash_value = 0
        
        # XOR in the player to move
        current_player = state.to_move()
        if current_player == 1:
            hash_value ^= _zobrist_player[0]  # Use global table
        else:
            hash_value ^= _zobrist_player[1]  # Use global table
        
        # XOR in all pieces on the board
        for position, weight in state.pieces.items():
            player_index = 0 if weight > 0 else 1
            piece_type = min(2, abs(weight) - 1)
            x, y = position
            hash_value ^= _zobrist_pieces[player_index][piece_type][x][y]
        
        return int(hash_value)
        
    def get(self, state, depth, alpha, beta):
        """
        Look up a position in the trans table.
        Returns: (value, best_move) if found and useful, None otherwise
        """
        # get hash
        hash_key = self.compute_hash(state)
        
        # Ensure the hash key is a Python int, not a numpy int
        if hasattr(hash_key, 'item'):
            hash_key = hash_key.item()  # Convert numpy types to native Python
        
        # if the hash is in the table 
        if hash_key in self.table:
            entry = self.table[hash_key]
            
            # Always count a hit whenever the position is found, regardless of usefulness
            self.hits += 1
            
            # Check if the entry is useful:
            # 1. Entry was searched at least as deep as we need now
            if entry['depth'] >= depth:
                if entry['flag'] == self.EXACT:
                    # Exact value is always useful
                    return entry['value'], entry['best_move']
                elif entry['flag'] == self.LOWER_BOUND and entry['value'] >= beta:
                    # Lower bound that causes a beta cutoff
                    return entry['value'], entry['best_move']
                elif entry['flag'] == self.UPPER_BOUND and entry['value'] <= alpha:
                    # Upper bound that causes an alpha cutoff
                    return entry['value'], entry['best_move']
        else:
            self.misses += 1 # add a miss
            
        return None, None # not found or useless
        
    def put(self, state, depth, value, flag, best_move):
        """
        Store a position in the transposition table.
        Uses replacement strategy to prefer deeper searches and exact values.
        """
        # Get the hash 
        hash_key = self.compute_hash(state)
        
        # Ensure the hash key is a Python int, not a numpy int
        if hasattr(hash_key, 'item'):
            hash_key = hash_key.item()  # Convert numpy types to native Python
        
        # Check if we should replace an existing entry
        replace = True
        if hash_key in self.table:
            old_entry = self.table[hash_key]
            
            # Don't replace deeper searches with shallower ones
            if old_entry['depth'] > depth:
                # Prefer exact values over bounds
                if old_entry['flag'] == self.EXACT and flag != self.EXACT:
                    replace = False
                # Otherwise only keep the deeper search
                else:
                    replace = False
        
        if replace:
            # Store entry
            self.table[hash_key] = {
                'depth': depth,
                'value': value,
                'flag': flag,
                'best_move': best_move
            }
        
        # If table is full, remove least valuable entries
        if len(self.table) >= self.max_size:
            # Simple strategy: Remove a random entry
            # More advanced would be to track entry age and replace oldest
            keys = list(self.table.keys())
            # Remove 10% of entries when we reach capacity
            num_to_remove = max(1, int(0.1 * self.max_size))
            for _ in range(num_to_remove):
                self.table.pop(random.choice(keys))
        
    def get_hit_rate(self):
        """Calculate the hit rate of the transposition table."""
        total = self.hits + self.misses
        if total > 0:
            return self.hits / total
        return 0
        
    def clear(self):
        """Clear the transposition table."""
        self.table.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self):
        """Get detailed statistics about the transposition table."""
        entry_types = {
            self.EXACT: 0,
            self.LOWER_BOUND: 0, 
            self.UPPER_BOUND: 0
        }
        depths = {}
        
        # Count entry types and depths
        for entry in self.table.values():
            entry_types[entry['flag']] += 1
            depth = entry['depth']
            depths[depth] = depths.get(depth, 0) + 1
            
        return {
            'size': len(self.table),
            'max_size': self.max_size,
            'hit_rate': self.get_hit_rate(),
            'hits': self.hits,
            'misses': self.misses,
            'entry_types': entry_types,
            'depths': depths
        }
    
    def print_stats(self):
        """Print detailed statistics about the transposition table."""

        if not DEBUG_MODE:
            print("DEBUG mode deactivated")
            return
        stats = self.get_stats()
        
        print("\n==== Transposition Table Statistics ====")
        print(f"Size: {stats['size']}/{stats['max_size']} ({stats['size']/stats['max_size']*100:.1f}% full)")
        print(f"Hit rate: {stats['hit_rate']:.2%} ({stats['hits']} hits, {stats['misses']} misses)")
        
        # Print entry types
        print("\nEntry Types:")
        types_labels = {
            self.EXACT: "EXACT",
            self.LOWER_BOUND: "LOWER_BOUND", 
            self.UPPER_BOUND: "UPPER_BOUND"
        }
        for flag, count in stats['entry_types'].items():
            print(f"  {types_labels[flag]}: {count} ({count/max(1, stats['size'])*100:.1f}%)")
            
        # Print depth distribution (at most 5 entries to avoid clutter)
        print("\nDepth Distribution (top 5):")
        sorted_depths = sorted(stats['depths'].items(), key=lambda x: x[1], reverse=True)
        for depth, count in sorted_depths[:5]:
            print(f"  Depth {depth}: {count} entries ({count/max(1, stats['size'])*100:.1f}%)")
            
        print("=======================================\n")

    def get_entry(self, state):
        hash_key = self.compute_hash(state)
        
        # Ensure the hash key is a Python int, not a numpy int
        if hasattr(hash_key, 'item'):
            hash_key = hash_key.item()  # Convert numpy types to native Python
            
        return self.table.get(hash_key)