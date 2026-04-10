import math
import time
import random

from agents.agent import Agent
from oxono.oxono import Game, State
Action = tuple[str, tuple[int, int], tuple[int, int]]


class AlphaBeta(Agent):
    """
    Alpha-Beta search agent using iterative deepening

    The agent combines several optimizations to improve search efficiency:
    - Alpha-beta pruning to reduce the search space
    - Iterative deepening for progressive deepening under time constraints
    - Time management with soft and hard bounds to control search duration
    - Transposition table with Zobrist hashing to reuse computed states
    - Move ordering (hash move, killer moves, heuristic evaluation)
    - Immediate loss pruning to avoid obvious tactical blunders
    - Tactical shortcuts like immediate win detection
    """

    def __init__(self, player):
        super().__init__(player)
        self.tt = {}
        self.tt_max_size = 100000 # still need to test what the max can be
        self.jack_the_ripper = {}
        self.soft_limit = 0; self.hard_limit = 0; self.start_time = 0; self.stop_search = False
        self.init_zobrist()

    def init_zobrist(self) -> None:
        """
        Initialize Zobrist hashing tables.

        Each possible element of the game state is assigned a random 64-bit integer:
        - Each (cell, symbol, player) combination
        - Each position of both totems
        - The current player

        These values are later combined using XOR to compute a unique hash
        for a given game state.
        """
        self.zobrist = {}
        self.zobrist_totem_O = {}
        self.zobrist_totem_X = {}
        for r in range(6):
            for c in range(6):
                for symbol in ["x", "o"]:
                    for player in [0, 1]:
                        self.zobrist[(r, c, symbol, player)] = random.getrandbits(64)
                self.zobrist_totem_O[(r, c)] = random.getrandbits(64)
                self.zobrist_totem_X[(r, c)] = random.getrandbits(64)
        self.zobrist_turn = random.getrandbits(64)

    def clear_tt(self) -> None:
        """
        Reduce the size of the transposition table when it exceeds its limit.
        The cleaning strategy progressively increases the minimum depth threshold to remove less reliable entries.
        If the table is still too large, a fallback mechanism removes random entries to ensure the size constraint is respected.
        """
        cleaning_depth = 1
        while len(self.tt) > self.tt_max_size and cleaning_depth <= 5:
            self.tt = {k: v for k, v in self.tt.items() if v[1] >= cleaning_depth}
            cleaning_depth += 1
        while len(self.tt) > self.tt_max_size:
            self.tt.pop(random.choice(list(self.tt.keys())))

    def tt_lookup(self, state: State, depth: int) -> tuple[float | None, Action | None]:
        """
        Retrieve a stored value and best move from the transposition table if available.

        Args:
            state (State): Current game state.
            depth (int): Required search depth.

        Returns:
            tuple[float | None, Action | None]:
                - value (float | None): Stored evaluation if found at sufficient depth.
                - move (Action | None): Best move associated with the stored state.
        """
        key = self.hash_state(state)
        if key in self.tt:
            val, stored_depth, move = self.tt[key]
            if stored_depth >= depth:
                return val, move
        return None, None

    def tt_store(self, state: State, value: float, depth: int, move: Action | None) -> None:
        """
        Store a state evaluation and best move in the transposition table
        The entry is stored only if it is new or computed at an equal or greater depth
        A greater depth means the value is more reliable

        Args:
            state (State): Current game state
            value (float): Evaluation value of the state
            depth (int): Depth at which the value was computed
            move (Action | None): Best move found from this state

        Returns:
            None
        """
        key = self.hash_state(state)
        if key not in self.tt or depth >= self.tt[key][1]:
            if len(self.tt) >= self.tt_max_size:
                self.clear_tt()
            self.tt[key] = (value, depth, move)

    def hash_state(self, state: State) -> int:
        """
        Compute the Zobrist hash of a game state.

        Args:
            state (State): Current game state.

        Returns:
            int: 64-bit hash representing the state.
        """
        h = 0
        for r in range(6):
            for c in range(6):
                cell = state.board[r][c]
                if cell is not None:
                    symbol, player = cell
                    h ^= self.zobrist[(r, c, symbol, player)]
        h ^= self.zobrist_totem_O[state.totem_O]
        h ^= self.zobrist_totem_X[state.totem_X]
        if state.current_player == 1:
            h ^= self.zobrist_turn
        return h

    def set_time_policy(self, state: State, remaining_time: float) -> None:
        """
        Set time limits for the current move based on remaining time and game progress.

        Args:
            state (State): Current game state.
            remaining_time (float): Remaining time (in seconds).

        Returns:
            None
        """
        ply = sum(cell is not None for row in state.board for cell in row)
        C = 1; MaxPly = 34
        self.start_time = time.time()
        self.soft_limit = remaining_time / (C + max(MaxPly - ply, 0))
        self.hard_limit = self.soft_limit * 2
        if self.shark_attack(state, self.player):
            self.soft_limit *= 0.5

    def check_timeout(self) -> None:
        """
        Check whether the allotted search time has been exceeded.

        Raises:
            TimeoutError: If the elapsed time exceeds the limit.
        """
        elapsed = time.time() - self.start_time
        if elapsed > self.hard_limit:
            raise TimeoutError()
        if elapsed > self.soft_limit:
            self.stop_search = True

    def shark_attack(self, state: State, player: int) -> Action | None:
        """
        Detect an immediate winning move (ply 1)

        Args:
            state (State): Current game state
            player (int): Player to play (0 or 1)

        Returns:
            tuple[str, tuple[int, int], tuple[int, int]] | None: Winning action if found, otherwise None
        """
        temp_state = state.copy()
        temp_state.current_player = player
        for action in Game.actions(temp_state):
            new_state = temp_state.copy()
            Game.apply(new_state, action)
            if Game._last_piece_won(new_state):
                return action
        return None

    def quick_eval(self, state: State, action: Action) -> float:
        """
        Quickly evaluate the result of applying an action to a state.

        Args:
            state (State): Current game state.
            action (Action): Action to simulate.

        Returns:
            float: Heuristic value of the resulting state.
        """
        new_state = state.copy()
        Game.apply(new_state, action)
        return self.evaluate(new_state)

    def immediate_loss_pruning(self, state: State, actions: list[Action]) -> list[Action]:
        """
        Remove actions that allow an immediate win for the opponent.

        Args:
            state (State): Current game state.
            actions (list[Action]): List of possible actions.

        Returns:
            list[Action]: Filtered list of safe actions.
        """
        safe_actions = []
        for a in actions:
            new_state = state.copy()
            Game.apply(new_state, a)
            if self.shark_attack(new_state, 1 - self.player) is None:
                safe_actions.append(a)
        return safe_actions

    def store_killer(self, depth: int, move: Action) -> None:
        """
        Store a killer move for a given search depth in a FIFO list.
        A killer move is a move that previously caused a cutoff at the same depth

        Args:
            depth (int): Current remaining depth in the search tree
            move (Action): Move that caused a cutoff

        Returns:
            None
        """
        if depth not in self.jack_the_ripper:
            self.jack_the_ripper[depth] = []
        killers = self.jack_the_ripper[depth]
        if move in killers:
            return
        killers.insert(0, move)
        if len(killers) > 3:
            killers.pop()

    def move_ordering(self, state: State, actions: list[Action], depth: int, reverse: bool) -> list[Action]:
        """
        Order actions to improve alpha-beta pruning

        Priority:
        1. Hash move (from transposition table)
        2. Killer moves
        3. Immediate loss pruning (safe moves)
        4. Heuristic evaluation

        Args:
            state (State): Current game state
            actions (list[Action]): List of possible actions
            depth (int): Current remaining depth
            reverse (bool): Determines the sorting direction for heuristic ordering ONLY
                True for max nodes (descending order)
                False for min nodes (ascending order)

        Returns:
            list[Action]: Ordered list of actions
        """
        ordered_actions = []
        _, tt_move = self.tt_lookup(state, depth)
        if tt_move in actions:
            actions.remove(tt_move)
            ordered_actions.append(tt_move)

        if depth in self.jack_the_ripper:
            for killer in self.jack_the_ripper[depth]:
                if killer in actions:
                    actions.remove(killer)
                    ordered_actions.append(killer)

        safe_actions = self.immediate_loss_pruning(state, actions)
        if safe_actions:
            actions = safe_actions

        scored = []
        for a in actions:
            score = self.quick_eval(state, a)
            scored.append((score, a))

        scored.sort(key=lambda x: x[0], reverse=reverse)
        ordered_actions.extend(a for _, a in scored)
        return ordered_actions

    def act(self, state: State, remaining_time: float) -> Action:
        """
        Select the best action using iterative deepening and alpha-beta search.

        Args:
            state (State): Current game state.
            remaining_time (float): Remaining time (in seconds).

        Returns:
            tuple[str, tuple[int, int], tuple[int, int]]:
                A move represented as (totem, totem_pos, piece_pos), where:
                - totem (str): The totem being moved ('O' or 'X').
                - totem_pos (tuple[int, int]): Target position of the totem (row, col).
                - piece_pos (tuple[int, int]): Position where the piece is placed (row, col).
        """
        if (move := self.shark_attack(state, self.player)) is not None: return move
        if (move := self.shark_attack(state, 1 - self.player)) is not None: return move

        depth = 1; self.stop_search = False
        best_move = self.move_ordering(state, Game.actions(state), depth=0, reverse=True)[0]
        self.set_time_policy(state, remaining_time)
        try:
            while not self.stop_search:
                self.check_timeout()
                move = self.alpha_beta(state, depth)
                if move is not None:
                    best_move = move
                depth += 1
        except TimeoutError:
            pass
        #print(f"[INFO] depth reached: {depth}")
        return best_move

    def alpha_beta(self, state: State, depth: int) -> Action:
        """
        Perform alpha-beta search up to a given depth.

        Args:
            state (State): Current game state.
            depth (int): Maximum search depth.
            start_time (float): Time at which the search started.
            time_limit (float): Maximum allowed search time.

        Returns:
            tuple[str, tuple[int, int], tuple[int, int]]:
                Best action found at this depth.
        """
        _, move = self.max_value(state, -math.inf, math.inf, depth)
        return move
      
    def max_value(self, state: State, alpha: float, beta: float, depth: int) -> tuple[float, Action | None]:
        """
        Compute the maximum value for the current player.

        Args:
            state (State): Current game state.
            alpha (float): Best value achievable by the maximizing player so far.
            beta (float): Best value achievable by the minimizing player so far.
            depth (int): Remaining search depth.
            start_time (float): Search start time.
            time_limit (float): Allowed search time.

        Returns:
            tuple[float, action | None]:
                - value (float): Evaluation of the state.
                - action (tuple or None): Best action leading to this value.
        """
        self.check_timeout()
        if Game.is_terminal(state):
            return Game.utility(state, self.player), None
        if depth == 0:
            return self.evaluate(state), None
        
        tt_val, tt_move = self.tt_lookup(state, depth)
        if tt_val is not None:
            return tt_val, tt_move
        
        v = -math.inf; move = None
        actions = self.move_ordering(state, Game.actions(state), depth, reverse=True)
        
        for a in actions:
            new_state = state.copy()
            Game.apply(new_state, a)
            v2, _ = self.min_value(new_state, alpha, beta, depth-1)
            if v2 > v :
                v, move = v2, a
                alpha = max(alpha, v)
            if v >= beta:
                self.store_killer(depth, a)
                self.tt_store(state, v, depth, move)
                return v, move
            
        self.tt_store(state, v, depth, move)
        return v, move

    def min_value(self, state: State, alpha: float, beta: float, depth: int) -> tuple[float, Action | None]:
        """
        Compute the minimum value for the opponent.

        Args:
            state (State): Current game state.
            alpha (float): Best value achievable by the maximizing player so far.
            beta (float): Best value achievable by the minimizing player so far.
            depth (int): Remaining search depth.
            start_time (float): Search start time.
            time_limit (float): Allowed search time.

        Returns:
            tuple[float, action | None]:
                - value (float): Evaluation of the state.
                - action (tuple or None): Best action leading to this value.
        """
        self.check_timeout()
        if Game.is_terminal(state):
            return Game.utility(state, self.player), None
        if depth == 0:
            return self.evaluate(state), None
        
        tt_val, tt_move = self.tt_lookup(state, depth)
        if tt_val is not None:
            return tt_val, tt_move
        
        v = math.inf; move = None
        actions = self.move_ordering(state, Game.actions(state), depth, reverse=False)

        for a in actions: 
            new_state = state.copy()
            Game.apply(new_state, a)
            v2, _ = self.max_value(new_state, alpha, beta, depth-1)
            if v2 < v :
                v, move = v2, a
                beta = min(beta, v)
            if v <= alpha:
                self.store_killer(depth, a)
                self.tt_store(state, v, depth, move)
                return v, move
                    
        self.tt_store(state, v, depth, move)
        return v, move

    def evaluate(self, state: State) -> int:
        """
        Heuristic evaluation of a non-terminal state.
        The score is based on alignments of symbols and pieces of the same player.
        
        STILL NEED IMPROVEMENT

        Args:
            state (State): Current game state.

        Returns:
            int: Evaluation score (positive if favorable, negative otherwise).
        """
        board = state.board
        score = 0
        directions = [(1, 0), (0, 1)]  # vertical, horizontal
        for r in range(6):
            for c in range(6):
                if board[r][c] is None:
                    continue
                symbol, player = board[r][c]
                for dr, dc in directions:
                    count_symbol = 0
                    count_color = 0
                    for i in range(4):
                        nr, nc = r + i*dr, c + i*dc
                        if not (0 <= nr < 6 and 0 <= nc < 6):
                            break
                        cell = board[nr][nc]
                        if cell is None: break
                        sym, pl = cell
                        if sym == symbol:
                            count_symbol += 1
                        else: break
                    for i in range(4):
                        nr, nc = r + i*dr, c + i*dc
                        if not (0 <= nr < 6 and 0 <= nc < 6): break
                        cell = board[nr][nc]
                        if cell is None: break
                        sym, pl = cell
                        if pl == player and sym != symbol:
                            score += 50
                        if pl == player and sym == symbol:
                            score += 20
                        else: break
                    # scoring
                    if player == self.player:
                        score += count_color * 5
                        score += count_symbol * 3
                    else:
                        score -= count_color * 5
                        score -= count_symbol * 3
        return score