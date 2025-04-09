
from numpy import random, uint64
from consts import *

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

    _zobrist_pieces = None # have to get zobrist values class-level
    _zobrist_player = None
    
    def __init__(self, max_size=10000):
        if TranspositionTable._zobrist_pieces is None:
            self._init_zobrist_tables()
        self.max_size = max_size
        self.table = {}
        self.hits = 0
        self.misses = 0

    def _init_zobrist_tables(self):
        rng = random.default_rng(seed=42)  # fixed seed for reproducibility
        TranspositionTable._zobrist_pieces = rng.integers(0, 2**64, size=(2, 3, 7, 8), dtype=uint64)
        TranspositionTable._zobrist_player = rng.integers(0, 2**64, size=2, dtype=uint64)
        
    def compute_hash(self, state):
        """
        Compute Zobrist hash for a game state.
        Returns the Zobrist hash value
        """
        hash_value = 0
        
        # XOR in the player to move
        if state.player == 1:
            hash_value ^= self.zobrist_player[0]
        else :
            hash_value ^= self.zobrist_player[1]
        
        # XOR in all pieces on the board
        for piece in state.pieces:
            player_index = 0 if piece.player == 1 else 1
            hash_value ^= self.zobrist_pieces[player_index][piece.type][piece.position[0]][piece.position[1]]
            
        return hash_value
        
    def get(self, state, depth, alpha, beta):
        """
        get a position in the trans table.
        Returns: (value, best_move) if found and useful, None otherwise
        """
        # get hash
        hash_key = self.compute_hash(state)
        
        # if the hash is in the table 
        if hash_key in self.table:
            a = self.table[hash_key]
            self.hits += 1 # add a hit
            
            # Check if the entry is useful
            if a['depth'] >= depth:
                if a['flag'] == self.EXACT:
                    return a['value'], a['best_move']
                elif a['flag'] == self.LOWER_BOUND and a['value'] >= beta:
                    return a['value'], a['best_move']
                elif a['flag'] == self.UPPER_BOUND and a['value'] <= alpha:
                    return a['value'], a['best_move']
        else:
            self.misses += 1 # add a miss
            
        return None, None # not found or useless
        
    def put(self, state, depth, value, flag, best_move):
        """
        Store a position in the transposition table.
        """
        # get the hash 
        hash_key = self.compute_hash(state)
        
        # if table is full, remove a random entry -> can be optimized ?
        if len(self.table) >= self.max_size:
            self.table.pop(random.choice(list(self.table.keys())))
            
        # store entry
        self.table[hash_key] = {
            'depth': depth,
            'value': value,
            'flag': flag,
            'best_move': best_move
        }
        
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