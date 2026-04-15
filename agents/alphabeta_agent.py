import math
import time
import random
import sys

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
    - Tactical shortcuts like immediate win detection
    """

    def __init__(self, player, debug=False, log_file="AB_log.log", tt_size=10000):
        super().__init__(player)
        self.agent_name = "Alpha Beta agent"
        self.debug = debug
        self.log_file =log_file
        self.tt = {}
        self.tt_size = tt_size # still need to test what the max can be
        self.jack_the_ripper = {}
        self.soft_limit = 0; self.hard_limit = 0; self.start_time = 0; self.stop_search = False
        self.init_zobrist()
        self.lines_of_4 = self.get_lines_of_4()

        self.tt_hits = 0
        self.tt_miss = 0
        self.tt_time = 0

    def log(self, msg):
        if self.debug :
            if self.log_file is not None :
                with open(self.log_file, "a") as f:
                    f.write(f"{msg}, {self.agent_name}" + "\n")
            print(f"{msg}, {self.agent_name}", file=sys.stderr)

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
        self.log(f"TT clearing started, len = {len(self.tt)}")
        cleaning_depth = 1
        while len(self.tt) > self.tt_size and cleaning_depth <= 5:
            self.tt = {k: v for k, v in self.tt.items() if v[1] >= cleaning_depth}
            cleaning_depth += 1
        while len(self.tt) > self.tt_size:
            self.tt.pop(random.choice(list(self.tt.keys())))
        self.log(f"TT clearing finished, len = {len(self.tt)}")

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
        t0 = time.perf_counter()

        key = self.hash_state(state)
        if key in self.tt:
            val, stored_depth, move = self.tt[key]
            if stored_depth >= depth:
                self.tt_hits += 1
                self.tt_time += time.perf_counter() - t0
                return val, move

        self.tt_miss += 1
        self.tt_time += time.perf_counter() - t0
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
        t0 = time.perf_counter()
        if self.stop_search: return
        key = self.hash_state(state)
        if key not in self.tt or depth >= self.tt[key][1]:
            if len(self.tt) >= self.tt_size:
                self.clear_tt()
            self.tt[key] = (value, depth, move)
            self.log(f"TT stored, len = {len(self.tt)}")
            self.tt_time += time.perf_counter() - t0


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
        self.log(f"Time policy set to : {self.soft_limit}")
        if self.shark_attack(state, self.player):
            self.log("Time policy reduced : shark attack")
            self.soft_limit *= 0.5

    def check_timeout(self) -> None:
        """
        Check whether the allotted search time has been exceeded.

        Raises:
            TimeoutError: If the elapsed time exceeds the limit.
        """
        elapsed = time.time() - self.start_time
        if elapsed > self.hard_limit:
            self.log("Hard limit reached")
            raise TimeoutError()
        if elapsed > self.soft_limit:
            self.log("Soft limit reached")
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
        return self.score_alignments(new_state)

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
        self.log("Killer move stored")
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
            ordered_actions.append(tt_move)

        if depth in self.jack_the_ripper:
            for killer in self.jack_the_ripper[depth]:
                if killer in actions:
                    ordered_actions.append(killer)
        scored = []
        for a in actions:
            if a not in ordered_actions :
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
        if (move := self.shark_attack(state, self.player)) is not None: 
            self.log("Winning move detected, playing it right now")
            return move
        if (move := self.shark_attack(state, 1 - self.player)) is not None:
            self.log("Imminent threat detected, blocking it right now")
            return move

        depth = 1
        best_move = self.move_ordering(state, Game.actions(state), depth, reverse=True)[0]
        self.set_time_policy(state, remaining_time)
        try:
            while True:
                self.stop_search = False
                move = self.alpha_beta(state, depth)
                if move is not None and not self.stop_search:
                    best_move = move
                if self.stop_search: break
                depth += 1
        except TimeoutError:
            pass
        self.log(f"[INFO] max depth reached: {depth}")
        self.log(f"TT hits: {self.tt_hits}, miss: {self.tt_miss}")
        self.log(f"TT time: {self.tt_time:.4f}s")
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
        actions = Game.actions(state)
        
        for a in actions:
            if self.stop_search: break

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
        actions = Game.actions(state)

        for a in actions: 
            if self.stop_search: break

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

    def get_lines_of_4(self, n: int = 6, k: int = 4) -> list[list[tuple[int, int]]]:
        """
        Generate all horizontal and vertical segments of length k on an n x n board.

        Args:
            n (int): Board size.
            k (int): Segment length.

        Returns:
            list[list[tuple[int, int]]]: List of segments of k positions (row, col).
        """
        lines = []
        for r in range(n):
            for c in range(n - k + 1):
                lines.append([(r, c + i) for i in range(k)])
        for c in range(n):
            for r in range(n - k + 1):
                lines.append([(r + i, c) for i in range(k)])
        return lines
    
    def evaluate(self, state: State) -> int:
        """
        Evaluate a game state from the perspective of self.player
        States are scored using heuristic components
        Current implementation relies on alignment-based evaluation, but the structure allows adding more features

        Args:
            state (State): current game state

        Returns:
            int: positive if favorable, negative if unfavorable
        """
        if self.shark_attack(state, state.current_player) is not None:
            return +100000 if state.current_player == self.player else -100000
        score = 0
        score += self.score_alignments(state)
        BONUS = 300
        last_player = 1 - state.current_player
        #for totem in ['O', 'X']:
        #    if self.super_totem(state, totem):
        #        score += BONUS if last_player == self.player else -BONUS
        return score

    def score_alignments(self, state: State) -> int:
        """
        Evaluate all segments of 4 cells and assign a heuristic score

        The evaluation is always from the perspective of self.player

        Scoring logic:
        - 3 aligned:
            - Uniform (same color + same symbol): strongest
            - Color alignment: strong (independent of symbol)
            - Symbol alignment: weaker (often exploitable by both players)
            - Depends on current_player (who can play next)

        - 2 aligned:
            - Open (both ends free): strong potential
            - Semi-open or basic: moderate potential
            - Color > Symbol

        Heuristic principles:
        - Prioritize immediate wins and threats
        - Penalize opponent threats more than rewarding own opportunities
        - Favor patterns exploitable by a single player
        - Reduce value of ambiguous (shared) patterns

        Returns:
            int: positive if favorable, negative if dangerous
        """
        ATTACK = 1; DEFENSE = 2.5
        THREE_UNIFORM = 15000; THREE_COLOUR  = 10000; THREE_SYMBOL  = 1000
        OPEN_TWO_COLOUR = 2500; OPEN_TWO_SYMBOL = 1000
        TWO_COLOUR = 500; TWO_SYMBOL = 150
        score = 0

        for line in self.lines_of_4:
            current_player = state.current_player
            cells = [state.board[r][c] for (r, c) in line]
            nb_empty = sum(1 for c in cells if c is None)
            nb_mine = sum(1 for c in cells if c is not None and c[1] == self.player)
            nb_enemy = sum(1 for c in cells if c is not None and c[1] == 1 - self.player)
            nb_x = sum(1 for c in cells if c is not None and c[0] == 'x')
            nb_o = sum(1 for c in cells if c is not None and c[0] == 'o')
            symbols = [None if c is None else c[0] for c in cells]
            colours = [None if c is None else c[1] for c in cells]

            if nb_empty == 1:
                if nb_mine == 3 and (nb_x == 3 or nb_o == 3):
                    score += THREE_UNIFORM if current_player == self.player else -THREE_UNIFORM * DEFENSE
                elif nb_mine == 3:
                    score += THREE_COLOUR if current_player == self.player else -THREE_COLOUR * DEFENSE
                elif nb_enemy == 3:
                    score -= THREE_COLOUR * DEFENSE if current_player == self.player else -THREE_COLOUR
                elif nb_x == 3 or nb_o == 3:
                    score += THREE_SYMBOL if current_player == self.player else -THREE_SYMBOL * DEFENSE

            if nb_empty == 2:
                if colours == [None, colours[1], colours[1], None] and colours[1] is not None:
                    score += OPEN_TWO_COLOUR * ATTACK if colours[1] == self.player else -OPEN_TWO_COLOUR * DEFENSE
                if symbols == [None, symbols[1], symbols[1], None] and symbols[1] is not None:
                    score -= OPEN_TWO_SYMBOL * DEFENSE
                if nb_mine == 2:
                    score += TWO_COLOUR * ATTACK
                elif nb_enemy == 2:
                    score -= TWO_COLOUR * DEFENSE
                elif nb_x == 2 or nb_o == 2:
                    score -= TWO_SYMBOL * DEFENSE
        return int(score)
    
    def super_totem(self, state: State, totem: str) -> bool:
        """
        Check whether a totem is locally trapped (surrounded on all four adjacent sides)
        This allows the player to place a piece anywhere on the board, making it sometimes a huge advantage 

        Args:
            state (State): The current game state
            totem (str): The totem to evaluate ('O' or 'X')

        Returns:
            bool: True if the totem is locally surrounded (no adjacent free squares),
                False otherwise
        """
        board = state.board
        r, c = state.totem_O if totem == 'O' else state.totem_X
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if not ((0 <= nr < 6) and (0 <= nc < 6)):
                continue
            if (board[nr][nc] is None):
                return False
        return True
