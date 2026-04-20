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

class AB7(Agent):
    """
    AB7
    This agent score 80 percent on inginious symetric (black and pink 4/5 games won)

    Black : 4/5 won 
        Game 0: won
        Game 1: won 
        Game 2: lost due to 4 threat did block one but won next turn to the second already there
                on turn 20pink played totem x but playing totem o would have been better preventing 
                black to create a double treath
        Game 3: won
        Game 4: lost
    Pink : 4/5 won
        Game 5: won
        Game 6: lost turn 20 here turn 17 black played totem O but it allow pink to creat a threat nex turn
                forcing us to block it turn 19 but the blocking creat another symbol treath that pink exploit and win turn 20
        Game 7: won
        Game 8: won
        Game 9: won


    """


    def __init__(self, player, debug=True, log_file="AB7_log.log", tt_size=100_000):
        super().__init__(player)
        self.tt = OrderedDict()
        self.tt_size = tt_size # still need to test what the max can be
        self.init_zobrist()
        self.stop_search = False

        # Performance and Debugging variables
        self.agent_name = "AB7 agent"
        self.debug = debug
        self.log_file =log_file
        self.tt_hits = 0; self.tt_miss = 0; self.tt_time = 0

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
        if h not in self.tt:
            return None, None, alpha, beta

        val, stored_depth, move, flag = self.tt[h] 
        self.tt.move_to_end(h) # moving to end, LRU cache first = oldest

        if stored_depth < depth:
            self.tt_miss += 1
            return None, move, alpha, beta
        

        if flag == EXACT:
            self.tt_hits += 1
            return val, move, alpha, beta

        elif flag == LOWERBOUND:
            if val >= beta :
                self.tt_hits += 1
                return val, move, alpha, beta
            alpha = max(alpha, val)
        elif flag == UPPERBOUND:
            if val <= alpha :
                self.tt_hits += 1
                return val, move, alpha, beta
            beta = min(beta, val)

        self.tt_miss += 1
        return None, move, alpha, beta

    def tt_store(self, h: int, value: float, depth: int, move: Action | None, flag: int) -> None:
        if h in self.tt:
            old_val, old_depth, _, old_flag = self.tt[h]
            if depth < old_depth:
                return
            if depth == old_depth:
                if old_flag == EXACT and flag != EXACT: # overwriting with exact value bc its better
                    return
            self.tt.pop(h) 
        elif len(self.tt) >= self.tt_size:
            self.tt.popitem(last=False) # LRU / FIFO
        self.log(f"Store: player={self.player}, flag={flag}, val={value}, depth={depth}")
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
        temp_state.current_player = player # for threat detection player is not the same for win detection its the same 
        for action in Game.actions(temp_state):
            new_state = temp_state.copy()
            Game.apply(new_state, action)
            if Game._last_piece_won(new_state) and self.pieces_in_stock(state, player, action):
                return action
        return None
    
    def pieces_in_stock(self, state: State, player: int, action: Action) -> bool:
        """Basicly not usefull if it return False we are cooked because we loose next turn 
            but may prolongue the game and avoid loosing by invalid action error"""
        if state.current_player == player : # not usefull if its win detection move always valid
            return True
        symbol = action[0]
        if symbol == 'X':
            pieces_left = state.pieces_x[player]
        else :
            pieces_left = state.pieces_o[player]
        return pieces_left > 0
    

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
            self.log(f"Move : {move}")
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
        _, move = self.max_value(state, -math.inf, math.inf, depth)
        return move
      
    def max_value(self, state: State, alpha: float, beta: float, depth: int, h: int = None) -> tuple[float, Action | None]:
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
            if v2 > v : # fail soft, (keeping highest value)
                v, move = v2, a
            alpha = max(alpha, v)
            if v >= beta: # beta cutoff, fail high 
                self.tt_store(h, v, depth, move, LOWERBOUND)
                return v, move
        
        flag = UPPERBOUND if v <= original_alpha else EXACT # fail low
        self.tt_store(h, v, depth, move, flag)
        return v, move

    def min_value(self, state: State, alpha: float, beta: float, depth: int, h: int = None) -> tuple[float, Action | None]:
        self.check_timeout()
        if Game.is_terminal(state):
            return Game.utility(state, self.player), None
        if depth == 0:
            return self.qui_est_ce(state, alpha, beta), None
        if h is None:
            h = self.hash_state(state)

        actions = Game.actions(state); original_beta = beta 
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

            v2, _ = self.max_value(new_state, alpha, beta, depth-1, child_h)
            if v2 < v: # fail soft, (keeping lowest value)
                v, move = v2, a
            beta = min(beta, v)
            if v <= alpha: # alpha cutoff, fail low
                self.tt_store(h, v, depth, move, UPPERBOUND)
                return v, move
        flag = LOWERBOUND if v >= original_beta else EXACT # fail high
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
        

    def qui_est_ce(self, state: State, alpha, beta):
        self.check_timeout()
        if Game.is_terminal(state):
            return Game.utility(state, self.player)

        stand_pat = self.evaluate(state) # value if we do nothing (quiesence convention)
        maximizing = (state.current_player == self.player)

        # checking if need need to explore or if the actual position can perform a cutoff + winding down the window search 
        if maximizing:
            if stand_pat >= beta:
                return stand_pat
            if stand_pat > alpha:
                alpha = stand_pat
            best_value = stand_pat
        else:
            if stand_pat <= alpha:
                return stand_pat
            if stand_pat < beta:
                beta = stand_pat
            best_value = stand_pat

        # while there is treath we continue the (basic) alpha beta search normaly until we find a stable state
        for a in Game.actions(state):
            new_state = state.copy()
            Game.apply(new_state, a)

            if not Game._last_piece_won(new_state) and self.shark_attack(new_state, new_state.current_player) is None: 
                continue

            score = self.qui_est_ce(new_state, alpha, beta)

            if maximizing:
                if score > best_value:
                    best_value = score
                alpha = max(alpha, score)
                if score >= beta:
                    return score
            else:
                if score < best_value:
                    best_value = score
                beta = min(beta, score)
                if score <= alpha:
                    return score

        return best_value