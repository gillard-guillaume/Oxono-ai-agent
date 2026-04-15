import math
import time
import random
import sys
from collections import OrderedDict

from agents.agent import Agent
from oxono.oxono import Game, State
Action = tuple[str, tuple[int, int], tuple[int, int]]

EXACT = 0
LOWERBOUND = 1
UPPERBOUND = 2

class AB6(Agent):
    """
    AB6 NOT WORKING 60 percent worse with pink
    """


    def __init__(self, player, debug=False, log_file="AB6_log.log", tt_size=100_000):
        super().__init__(player)
        self.agent_name = "AB6 agent"
        self.debug = debug
        self.log_file =log_file
        self.tt = OrderedDict()
        self.tt_size = tt_size # still need to test what the max can be
        self.init_zobrist()
        self.stop_search = False

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

    def tt_lookup(self, h: int, depth: int, alpha: float, beta: float) -> tuple[float | None, Action | None, float, float]:
        """
        Retrieve stored information from the transposition table and update alpha beta bounds if applicable

        Args:
            state (State): Current game state
            depth (int): Required search depth
            alpha (float): Current alpha value
            beta (float): Current beta value

        Returns:
            tuple[float | None, Action | None, float, float]:
                - value (float | None): Stored evaluation if usable
                - move (Action | None): Best move associated with the state
                - alpha (float): Possibly updated alpha
                - beta (float): Possibly updated beta
        """

        if h not in self.tt:
            return None, None, alpha, beta

        val, stored_depth, move, flag = self.tt[h]
        if stored_depth >= depth:
            self.tt.move_to_end(h) # If accessed but not usefull it's bad quality        

        if stored_depth < depth:
            self.tt_miss += 1
            return None, move, alpha, beta

        if flag == EXACT:
            self.tt_hits += 1
            return val, move, alpha, beta

        elif flag == LOWERBOUND:
            alpha = max(alpha, val)
        elif flag == UPPERBOUND:
            beta = min(beta, val)

        if alpha >= beta:
            self.tt_hits += 1
            return val, move, alpha, beta

        self.tt_miss += 1
        return None, move, alpha, beta

    def tt_store(self, h: int, value: float, depth: int, move: Action | None, flag: int) -> None:
        """
        Store a state evaluation in the transposition table
        The entry is stored only if it is new or computed at an equal or greater depth

        Args:
            state (State): Current game state
            value (float): Evaluation value of the state
            depth (int): Depth at which the value was computed
            move (Action | None): Best move found from this state
            flag (int): Type of bound EXACT, LOWERBOUND or UPPERBOUND

        Returns:
            None
        """

        if h in self.tt:
            if depth < self.tt[h][1]:
                return
            self.tt.pop(h) 
        elif len(self.tt) >= self.tt_size:
            self.tt.popitem(last=False) # LRU / FIFO
        self.tt[h] = (value, depth, move, flag)


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
    
    def update_hash(self, h: int, state: State, action: Action) -> int:
        totem, totem_pos, piece_pos = action
        if totem == 'O':
            h ^= self.zobrist_totem_O[state.totem_O]
            h ^= self.zobrist_totem_O[totem_pos]
            symbol = 'o'
        else:
            h ^= self.zobrist_totem_X[state.totem_X]
            h ^= self.zobrist_totem_X[totem_pos]
            symbol = 'x'

        r, c = piece_pos
        player = state.current_player
        h ^= self.zobrist[(r, c, symbol, player)]
        h ^= self.zobrist_turn
        return h   

    def move_ordering(self, state: State, actions: list[Action], depth: int, h: int, reverse: bool) -> list[Action]:
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
        scored = []
        tt_move = self.tt[h][2] if h in self.tt else None
        for a in actions:
            if a == tt_move:
                continue
            h_child = self.update_hash(h, state, a)
            score = None
            if h_child in self.tt:
                val, d, _, flag = self.tt[h_child]
                if d >= depth - 1 and flag == EXACT:
                    score = val
            if score is None:
                new_state = state.copy()
                Game.apply(new_state, a)
                score = self.evaluate(new_state)
            scored.append((score, a))
        scored.sort(key=lambda x: x[0], reverse=reverse)
        if tt_move in actions:
            return [tt_move] + [a for _, a in scored]
        return [a for _, a in scored]

        
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
        C = 0.5; MaxPly = 34
        self.start_time = time.time()
        self.soft_limit = remaining_time / (C + max(MaxPly - ply, 0))
        self.hard_limit = self.soft_limit * 2
        self.log(f"Time policy set to : {self.soft_limit}")

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

    def is_critical(self, state: State):
        actions = Game.actions(state)
        for a in actions:
            new_state = state.copy()
            Game.apply(new_state, a)
            if Game._last_piece_won(new_state):
                return True
        return False

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
        self.set_time_policy(state, remaining_time)
        depth = 1; last_completed_depth = 0
        actions = Game.actions(state)
        best_move = actions[0]
        self.stop_search = False

        try:
            while True:
                move = self.alpha_beta(state, depth)
                if move is not None:
                    best_move = move
                    last_completed_depth += 1
                depth += 1
        except TimeoutError:
            pass
        self.log(f"[INFO] max depth reached: {last_completed_depth}")
        self.log(f"TT hits: {self.tt_hits}, miss: {self.tt_miss}")
        self.log(f"TT time: {self.tt_time:.4f}s")
        self.log(f"TT len = {len(self.tt)}")
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
      
    def max_value(self, state: State, alpha: float, beta: float, depth: int, h: int = None) -> tuple[float, Action | None]:
        """
        Compute the maximum value for the current player

        Args:
            state (State): Current game state
            alpha (float): Best value achievable by the maximizing player so far
            beta (float): Best value achievable by the minimizing player so far
            depth (int): Remaining search depth
            start_time (float): Search start time
            time_limit (float): Allowed search time

        Returns:
            tuple[float, action | None]:
                - value (float): Evaluation of the state
                - action (tuple or None): Best action leading to this value
        """
        self.check_timeout()
        if Game.is_terminal(state):
            return Game.utility(state, self.player), None
        if depth == 1 and self.is_critical(state):
            depth += 1
        if depth == 0:
            return self.evaluate(state), None
        if h is None:
            h = self.hash_state(state)
        actions = Game.actions(state)
        original_alpha = alpha
        tt_val, tt_move, alpha, beta = self.tt_lookup(h, depth, alpha, beta)

        if tt_val is not None:
            return tt_val, tt_move
        
        v, move = -math.inf, None
        actions = self.move_ordering(state, actions, depth, h, reverse=True)
        
        for a in actions:
            if self.stop_search:
                raise TimeoutError
            child_h = self.update_hash(h, state, a)
            new_state = state.copy()
            Game.apply(new_state, a)
            if self.debug :
                assert child_h == self.hash_state(new_state), f"Hash inconsistant pour action {a}"
            v2, _ = self.min_value(new_state, alpha, beta, depth-1, child_h)
            if v2 > v :
                v, move = v2, a
            alpha = max(alpha, v)
            if v >= beta:
                self.tt_store(h, v, depth, move, LOWERBOUND)
                return v, move
        
        flag = UPPERBOUND if v < original_alpha else EXACT
        self.tt_store(h, v, depth, move, flag)
        return v, move

    def min_value(self, state: State, alpha: float, beta: float, depth: int, h: int = None) -> tuple[float, Action | None]:
        """
        Compute the minimum value for the opponent

        Args:
            state (State): Current game state
            alpha (float): Best value achievable by the maximizing player so far
            beta (float): Best value achievable by the minimizing player so far
            depth (int): Remaining search depth
            start_time (float): Search start time
            time_limit (float): Allowed search time

        Returns:
            tuple[float, action | None]:
                - value (float): Evaluation of the state
                - action (tuple or None): Best action leading to this value
        """
        self.check_timeout()
        if Game.is_terminal(state):
            return Game.utility(state, self.player), None
        if depth == 1 and self.is_critical(state):
            depth += 1
        if depth == 0:
            return self.evaluate(state), None
        if h is None:
            h = self.hash_state(state)

        actions = Game.actions(state)    
        original_beta = beta
        tt_val, tt_move, alpha, beta = self.tt_lookup(h, depth, alpha, beta)

        if tt_val is not None:
            return tt_val, tt_move

        v, move = +math.inf, None
        actions = self.move_ordering(state, actions, depth, h, reverse=False)

        for a in actions:
            if self.stop_search:
                raise TimeoutError
            child_h = self.update_hash(h, state, a)
            new_state = state.copy()
            Game.apply(new_state, a)
            if self.debug :
                assert child_h == self.hash_state(new_state), f"Hash inconsistant pour action {a}"
            v2, _ = self.max_value(new_state, alpha, beta, depth-1, child_h)
            if v2 < v:
                v, move = v2, a
            beta = min(beta, v)
            if v <= alpha:
                self.tt_store(h, v, depth, move, UPPERBOUND)
                return v, move
        flag = LOWERBOUND if v > original_beta else EXACT
        self.tt_store(h, v, depth, move, flag)
        return v, move

    def evaluate(self, state: State) -> float:
        if Game.is_terminal(state):
            return Game.utility(state, self.player)

        score = 0
        board = state.board
        directions = [(1, 0), (0, 1)]

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
                        if cell is None:
                            break

                        sym, pl = cell

                        if sym == symbol:
                            count_symbol += 1
                        else:
                            break

                    for i in range(4):
                        nr, nc = r + i*dr, c + i*dc
                        if not (0 <= nr < 6 and 0 <= nc < 6):
                            break
                        cell = board[nr][nc]
                        if cell is None:
                            break

                        sym, pl = cell

                        if pl == player:
                            count_color += 1
                        else:
                            break

                    val = count_symbol * 3 + count_color * 5

                    if player == self.player:
                        score += val
                    else:
                        score -= val
        return math.tanh(score / 300)