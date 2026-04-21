import math
import time
import random
import sys
from collections import OrderedDict

from agents.agent import Agent
from oxono.oxono import Game, State
Action = tuple[str, tuple[int, int], tuple[int, int]]

EXACT = 0; LOWERBOUND = 1; UPPERBOUND = 2

class AB8(Agent):
    """
    Alpha-beta search agent with iterative deepening and transposition table

    Combines alpha-beta pruning, Zobrist hashing, move ordering, and quiescence search to efficiently explore the game tree
    
    Simple heuristic evaluation function
    """

    def __init__(self, player, debug=False, log_file="AB8_log.log", tt_size=100_000):
        super().__init__(player)
        random.seed(42) # for reproducibility
        self.tt = OrderedDict()
        self.tt_size = tt_size
        self.init_zobrist()
        self.stop_search = False
        # Performance and Debugging variables
        self.agent_name = "AB8 agent"
        self.debug = debug
        self.log_file = log_file
        self.tt_hits = 0; self.tt_miss = 0; self.tt_time = 0

    def log(self, msg: str) -> None:
        """
        Log a debug message to stderr and optionally to a file.

        Args:
            msg (str): Message to log.
        """
        if self.debug :
            if self.log_file is not None :
                with open(self.log_file, "a") as f:
                    f.write(f"{msg}, {self.agent_name}" + "\n")
            print(f"{msg}, {self.agent_name}", file=sys.stderr)

    def init_zobrist(self) -> None:
        """
        Initialize Zobrist hashing tables

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
    
    def hash_state(self, state: State) -> int:
        """
        Compute the Zobrist hash of a game state

        The hash is computed by "XOR-ing" precomputed random values associated with:
        - Occupied board cells (position, symbol, player)
        - Totem positions
        - Current player

        Collisions are theoretically possible but negligible in practice
        given the limited number of explored states

        Args:
            state (State): Current game state

        Returns:
            int: 64-bit hash representing the state
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
        """
        Update the Zobrist hash after applying an action

        The update follow the same logic as hash_state and avoid
        recomputing the hash from scratch

        Args:
            h (int): Current hash value
            state (State): Game state before applying the action
            action (Action): Action to apply

        Returns:
            int: Updated hash value
        """
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

    def tt_lookup(self, h: int, depth: int, alpha: float, beta: float) -> tuple[float | None, Action | None, float, float]:
        """
        Lookup a state in the transposition table and update search bounds

        If an entry exists with sufficient depth :
        - Return the value directly if it is EXACT of if a global cutoff occur
        - Update alpha or beta if it is a bound (LOWERBOUND or UPPERBOUND)

        Args:
            h (int): Zobrist hash of the state
            depth (int): Current search depth
            alpha (float): Current alpha bound
            beta (float): Current beta bound

        Returns:
            tuple[float | None, Action | None, float, float]:
                - value: Stored evaluation if usable, otherwise None
                - move: Stored best move for ordering
                - alpha: Updated alpha bound
                - beta: Updated beta bound
        """
        if h not in self.tt:
            return None, None, alpha, beta
        val, stored_depth, move, flag = self.tt[h] 
        if stored_depth >= depth:
            self.tt.move_to_end(h) # LRU update
        if stored_depth < depth:
            self.tt_miss += 1
            return None, move, alpha, beta
        if flag == EXACT:
            self.tt_hits += 1
            return val, move, alpha, beta # fail-soft, ok 
        elif flag == LOWERBOUND:
            alpha = max(alpha, val)
        elif flag == UPPERBOUND:
            beta = min(beta, val)
        # global cutoff, equal to beta cutoff in max and alpha cutoff in min
        if alpha >= beta :
            self.tt_hits += 1
            return val, move, alpha, beta # fail-soft, ok 
        self.tt_miss += 1
        return None, move, alpha, beta

    def tt_store(self, h: int, value: float, depth: int, move: Action | None, flag: int) -> None:
        """
        Store a state evaluation in the transposition table

        An entry is stored only if the state is new, or the new depth >= to the stored depth

        If the table is full, the least recently used entry is removed

        Args:
            h (int): Zobrist hash of the state
            value (float): Evaluation of the state
            depth (int): Search depth at which the value was computed
            move (Action | None): Best move associated with the state
            flag (int): Type of entry (EXACT, LOWERBOUND, UPPERBOUND)

        Returns:
            None
        """
        if h in self.tt:
            if depth < self.tt[h][1]:
                return
        elif len(self.tt) >= self.tt_size:
            self.tt.popitem(last=False) # LRU eviction
        self.log(f"Store: player={self.player}, flag={flag}, val={value}, depth={depth}")
        self.tt[h] = (value, depth, move, flag)

    def move_ordering(self, state: State, actions: list[Action], depth: int, h: int, reverse: bool) -> list[Action]:
        """
        Order actions to improve alpha-beta pruning

        Priority :
            1) The hash move from the transposition table (if available)
            2) Actions evaluated using transposition table values when possible
            3) Otherwise, heuristic evaluation of resulting states

        Args:
            state (State): Current game state
            actions (list[Action]): List of possible actions
            depth (int): Current remaining depth
            h (int): Zobrist hash of the current state
            reverse (bool): Sorting direction for heuristic scores
                True for max nodes (descending)
                False for min nodes (ascending)

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
        Detect an immediate winning move (one ply ahead)

        Args:
            state (State): Current game state
            player (int): Player to play (0 or 1)

        Returns:
            tuple[str, tuple[int, int], tuple[int, int]] | None: Winning action if found, otherwise None
        """
        temp_state = state.copy()
        temp_state.current_player = player # for threat detection, "skipping our turn"
        for action in Game.actions(temp_state):
            new_state = temp_state.copy()
            Game.apply(new_state, action)
            if Game._last_piece_won(new_state):
                return action
        return None
    
    def qui_est_ce(self, state: State, alpha: float, beta: float, q_depth: int = 0) -> float:
        """
        Quiescence search limited to tactical moves (immediate wins)
        
        Uses a stand pat evaluation and fail-hard alpha-beta pruning, which tends to provide more stable results in quiescence search
        The search is bounded by a maximum quiescence depth

        Args:
            state (State): Current game state
            alpha (float): Alpha bound
            beta (float): Beta bound
            q_depth (int): Current quiescence depth

        Returns:
            float: Evaluated score of the position
        """
        self.check_timeout()
        if Game.is_terminal(state):
            return Game.utility(state, self.player)

        maximizing = (state.current_player == self.player)
        static_eval = self.evaluate(state) # stand pat
        static_eval, alpha, beta, cutoff = self.quiesence_ab(static_eval, alpha, beta, maximizing)
        if cutoff : 
            return static_eval # fail hard so either alpha or beta
        best_value = static_eval
        if q_depth >= 4: 
            return best_value

        stratego = []
        for a in Game.actions(state):
            new_state = state.copy()
            Game.apply(new_state, a)
            if Game._last_piece_won(new_state):
                stratego.append(new_state)

        if not stratego: return best_value
        
        for tactical_state in stratego:
            score = self.qui_est_ce(tactical_state, alpha, beta, q_depth+1)
            value, alpha, beta, cutoff = self.quiesence_ab(score, alpha, beta, maximizing)
            if cutoff:
                return value # fail hard so either alpha or beta
            best_value = max(value, best_value) if maximizing else min(value, best_value)
        return best_value
        
    def quiesence_ab(self, value: float, alpha: float, beta: float, maximizing: bool) -> tuple[float, float, float, bool]:
        """
        Apply a fail-hard alpha-beta for quiescence search
        Updates alpha or beta based on the evaluated value and detects cutoffs
        Uses a fail-hard scheme, returning the bound on cutoff

        Args:
            value (float): Evaluated score
            alpha (float): Alpha bound
            beta (float): Beta bound
            maximizing (bool): True if maximizing node, False otherwise

        Returns:
            tuple[float, float, float, bool]:
                - value: Updated value (or bound on cutoff)
                - alpha: Updated alpha
                - beta: Updated beta
                - cutoff: True if pruning occurred
        """
        if maximizing:
            if value >= beta:
                return beta, alpha, beta, True # Fail hard (may remove some bug, source : some obscur forums from 2008)
            alpha = max(alpha, value)
        else:
            if value <= alpha:
                return alpha, alpha, beta, True # Fail hard (idem)
            beta = min(beta, value)
        return value, alpha, beta, False
    

    def set_time_policy(self, state: State, remaining_time: float) -> None:
        """
        Set time limits for the current move based on game progression

        Allocates time dynamically by estimating the number of remaining plies
        Defines a soft limit for iterative deepening and a hard limit for cutoff

        Args:
            state (State): Current game state
            remaining_time (float): Remaining time (seconds)

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
        Monitor search time and enforce time limits

        Raises a TimeoutError if the hard limit is exceeded
        Sets a stop flag if the soft limit is reached to stop search

        Raises:
            TimeoutError: If the hard time limit is exceeded
        """
        elapsed = time.time() - self.start_time
        if elapsed >= self.hard_limit:
            self.log("Hard limit reached")
            raise TimeoutError()
        if elapsed >= self.soft_limit:
            self.log("Soft limit reached")
            self.stop_search = True

    def act(self, state: State, remaining_time: float) -> Action:
        """
        Select an action using iterative deepening and alpha-beta search

        Immediate tactical situations are handled first
        Then iterative deepening is performed within allocated time keeping the best fully evaluated move

        Args:
            state (State): Current game state
            remaining_time (float): Remaining time (seconds)

        Returns:
            Action: Selected move (totem, totem_pos, piece_pos)
        """
        if (move := self.shark_attack(state, self.player)) is not None: 
            self.log("Winning move detected, playing it right now")
            return move
        if (move := self.shark_attack(state, 1 - self.player)) is not None:
            self.log("Imminent threat detected, blocking it right now"); self.log(f"Move : {move}")
            return move
        self.set_time_policy(state, remaining_time); self.stop_search = False
        depth = 1; last_completed_depth = 0
        actions = Game.actions(state); best_move = actions[0]

        try:
            while True:
                move = self.alpha_beta(state, depth)
                if not self.stop_search:
                    best_move = move; last_completed_depth = depth
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
        Run alpha-beta search to a given depth and return the best action

        Args:
            state (State): Current game state
            depth (int): Search depth

        Returns:
            Action: Best action found at this depth
        """
        _, move = self.max_value(state, -math.inf, math.inf, depth)
        return move
      
    def max_value(self, state: State, alpha: float, beta: float, depth: int, h: int = None) -> tuple[float, Action | None]:
        """
        Compute the maximum value for self.player using alpha-beta search

        Applies transposition table lookup, move ordering, and quiescence search at leaf nodes
        Stores results in the transposition table with appropriate flags

        Transposition table flag logic (max node):
            Let original_alpha be the initial alpha value
            - If a beta cutoff occurs: v >= beta → store LOWERBOUND
            - If no beta cutoff occurs, all children are explored but are these exact values ?
                Consider children values v_child from MIN nodes:
                - If a alpha cutoff occurred in a child (v_child <= alpha), then alpha remains unchanged so alpha == original_alpha
                - If no alpha cutoff occurred in a child (v_child > alpha), then v = v_child because v = max(v, v_child)
                    and alpha = v because alpha = max(old_alpha, v) and v_child > alpha
                    so we got alpha > original_alpha because old_alpha = original_alpha and since alpha = v, v > original_alpha if no cutoff
                - So if we got v <= original_alpha that means a cutoff did happen and since it was an alpha cutoff it's UPPERBOUND
                - If not then we now all the node are explored without cutoff so it's an EXACT value

        Args:
            state (State): Current game state
            alpha (float): Alpha bound
            beta (float): Beta bound
            depth (int): Remaining search depth
            h (int | None): Zobrist hash of the state (optional)

        Returns:
            tuple[float, Action | None]: Max value and associated action
        """
        self.check_timeout()

        if Game.is_terminal(state):
            return Game.utility(state, self.player), None
        if depth == 0:
            return self.qui_est_ce(state, alpha, beta), None
        if h is None:
            h = self.hash_state(state)

        actions = Game.actions(state); original_alpha = alpha
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
            v2, _ = self.min_value(new_state, alpha, beta, depth-1, child_h)
            if v2 > v : # v = max(v, v2)
                v, move = v2, a
            alpha = max(alpha, v)
            if v >= beta :
                self.tt_store(h, v, depth, move, LOWERBOUND)
                return v, move # fail-soft, fail-high
        # See min node for explaination
        flag = UPPERBOUND if v <= original_alpha else EXACT
        self.tt_store(h, v, depth, move, flag)
        return v, move # fail-soft, ok 

    def min_value(self, state: State, alpha: float, beta: float, depth: int, h: int = None) -> tuple[float, Action | None]:
        """
        Compute the minimum value for the opponent using alpha-beta search

        Applies transposition table lookup, move ordering, and quiescence search at leaf nodes 
        Stores results in the transposition table with appropriate flags

        Transposition table flag logic (min node), cfr max node same reasoning but inversed

        Args:
            state (State): Current game state
            alpha (float): Alpha bound
            beta (float): Beta bound
            depth (int): Remaining search depth
            h (int | None): Zobrist hash of the state (optional)

        Returns:
            tuple[float, Action | None]: Min value and associated action
        """
        self.check_timeout()
        if Game.is_terminal(state):
            return Game.utility(state, self.player), None
        if depth == 0:
            return self.qui_est_ce(state, alpha, beta), None
        if h is None:
            h = self.hash_state(state)

        actions = Game.actions(state); original_beta = beta 
        tt_val, tt_move, alpha, beta = self.tt_lookup(h, depth, alpha, beta)

        if tt_val is not None: return tt_val, tt_move

        v, move = +math.inf, None
        actions = self.move_ordering(state, actions, depth, h, reverse=False)
        for a in actions:
            if self.stop_search:
                raise TimeoutError
            child_h = self.update_hash(h, state, a)
            new_state = state.copy()
            Game.apply(new_state, a)

            v2, _ = self.max_value(new_state, alpha, beta, depth-1, child_h)
            if v2 < v: # v = min(v, v2)
                v, move = v2, a
            beta = min(beta, v)
            if v <= alpha :
                self.tt_store(h, v, depth, move, UPPERBOUND)
                return v, move # fail-low, fail-soft
        flag = LOWERBOUND if v >= original_beta else EXACT 
        self.tt_store(h, v, depth, move, flag)
        return v, move # fail-soft, ok

    def evaluate(self, state: State) -> float:
        """
        Simple heuristic evaluation of a game state

        Scores alignments of symbols and pieces for both players by scanning horizontal and vertical directions 
        The final score is normalized using a tanh function to match utility score
        Favouring colours over symbols

        Args:
            state (State): Current game state

        Returns:
            float: Evaluation score in [-1, 1]
        """
        if Game.is_terminal(state):
            return Game.utility(state, self.player)
        SYMBOLS = 3; COLOURS = 5
        score = 0; board = state.board; directions = [(1, 0), (0, 1)]
        for r in range(6):
            for c in range(6):
                if board[r][c] is None: continue
                symbol, player = board[r][c]
                for dr, dc in directions:
                    count_symbol = 0; count_color = 0
                    for i in range(4):
                        nr, nc = r + i*dr, c + i*dc
                        if not (0 <= nr < 6 and 0 <= nc < 6): break
                        cell = board[nr][nc]
                        if cell is None: break
                        sym, pl = cell
                        if sym == symbol: count_symbol += 1
                        if pl == player: count_color += 1
                        else: break
                    val = count_symbol * SYMBOLS + count_color * COLOURS
                    score += val if player == self.player else -val
        return math.tanh(score / 300)