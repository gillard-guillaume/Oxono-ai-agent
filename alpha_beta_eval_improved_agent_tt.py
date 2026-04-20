from agent import Agent
from oxono import Game
import math
import time
import random
from collections import OrderedDict


class TimeoutError(Exception):
    pass


class MyAgent(Agent):
    """
    Alpha-Beta agent with:
    - iterative deepening and time control
    - transposition table with Zobrist hashing
    - TT move + killer move ordering
    - more threat-sensitive evaluation
    """

    BOARD_SIZE = 6
    TT_MAX_SIZE = 100_000
    EXACT = 0
    LOWERBOUND = 1
    UPPERBOUND = 2

    def __init__(self, player):
        super().__init__(player)

        self._rng = random.Random(1337 + player)

        self.zobrist_board = [
            [[self._rng.getrandbits(64) for _ in range(4)] for _ in range(self.BOARD_SIZE)]
            for _ in range(self.BOARD_SIZE)
        ]
        self.zobrist_totem_x = [
            [self._rng.getrandbits(64) for _ in range(self.BOARD_SIZE)]
            for _ in range(self.BOARD_SIZE)
        ]
        self.zobrist_totem_o = [
            [self._rng.getrandbits(64) for _ in range(self.BOARD_SIZE)]
            for _ in range(self.BOARD_SIZE)
        ]
        self.zobrist_side = self._rng.getrandbits(64)
        self.zobrist_pieces_x = [[self._rng.getrandbits(64) for _ in range(9)] for _ in range(2)]
        self.zobrist_pieces_o = [[self._rng.getrandbits(64) for _ in range(9)] for _ in range(2)]

        self.tt = OrderedDict()
        self.killer_moves = {}

    def act(self, state, remaining_time):
        actions = list(Game.actions(state))
        if not actions:
            return None

        for action in actions:
            next_state = state.copy()
            Game.apply(next_state, action)
            if Game.is_terminal(next_state) and Game.utility(next_state, self.player) == 1:
                return action

        pieces_placed = 32 - (
            state.pieces_x[0] + state.pieces_x[1] + state.pieces_o[0] + state.pieces_o[1]
        )
        moves_left_estimate = max(1, 32 - pieces_placed)

        reserve = 20.0
        usable_time = max(0.05, remaining_time - reserve)
        base_budget = usable_time / moves_left_estimate

        if pieces_placed < 8:
            move_budget = min(2.0, 0.8 * base_budget)
        elif pieces_placed < 20:
            move_budget = min(4.0, 1.2 * base_budget)
        else:
            move_budget = min(8.0, 1.8 * base_budget)

        move_budget = max(0.05, min(move_budget, remaining_time * 0.25))
        deadline = time.perf_counter() + move_budget

        best_move = actions[0]
        depth = 1
        root_hash = self.compute_hash(state)

        while True:
            if time.perf_counter() >= deadline:
                break
            try:
                ordered_root_actions = self.ordered_actions(
                    state=state,
                    actions=actions,
                    maximizing=True,
                    ply=0,
                    tt_move=self.tt_best_move(root_hash),
                )
                value, move = self.max_value(
                    state=state,
                    depth=depth,
                    alpha=-math.inf,
                    beta=math.inf,
                    deadline=deadline,
                    ply=0,
                    state_hash=root_hash,
                    actions=ordered_root_actions,
                )
                if move is not None:
                    best_move = move
                depth += 1
            except TimeoutError:
                break

        return best_move

    def check_time(self, deadline):
        if time.perf_counter() >= deadline:
            raise TimeoutError

    def max_value(self, state, depth, alpha, beta, deadline, ply, state_hash, actions=None):
        self.check_time(deadline)
        alpha_orig = alpha
        beta_orig = beta

        if Game.is_terminal(state):
            return self.terminal_score(state, depth), None

        if depth == 0:
            return self.evaluate(state), None

        tt_entry = self.tt_lookup(state_hash, depth)
        if tt_entry is not None:
            value, flag, best_move = tt_entry
            if flag == self.EXACT:
                return value, best_move
            if flag == self.LOWERBOUND:
                alpha = max(alpha, value)
            elif flag == self.UPPERBOUND:
                beta = min(beta, value)
            if alpha >= beta:
                return value, best_move
            tt_move = best_move
        else:
            tt_move = None

        if actions is None:
            actions = self.ordered_actions(
                state=state,
                actions=list(Game.actions(state)),
                maximizing=True,
                ply=ply,
                tt_move=tt_move,
            )

        best_value = -math.inf
        best_move = None

        for action in actions:
            self.check_time(deadline)
            next_state = state.copy()
            Game.apply(next_state, action)
            next_hash = self.compute_hash(next_state)

            value, _ = self.min_value(
                next_state, depth - 1, alpha, beta, deadline, ply + 1, next_hash
            )

            if value > best_value:
                best_value = value
                best_move = action

            alpha = max(alpha, best_value)
            if alpha >= beta:
                self.add_killer_move(ply, action)
                break

        flag = self.EXACT
        if best_value <= alpha_orig:
            flag = self.UPPERBOUND
        elif best_value >= beta_orig:
            flag = self.LOWERBOUND

        self.tt_store(state_hash, depth, best_value, flag, best_move)
        return best_value, best_move

    def min_value(self, state, depth, alpha, beta, deadline, ply, state_hash, actions=None):
        self.check_time(deadline)
        alpha_orig = alpha
        beta_orig = beta

        if Game.is_terminal(state):
            return self.terminal_score(state, depth), None

        if depth == 0:
            return self.evaluate(state), None

        tt_entry = self.tt_lookup(state_hash, depth)
        if tt_entry is not None:
            value, flag, best_move = tt_entry
            if flag == self.EXACT:
                return value, best_move
            if flag == self.LOWERBOUND:
                alpha = max(alpha, value)
            elif flag == self.UPPERBOUND:
                beta = min(beta, value)
            if alpha >= beta:
                return value, best_move
            tt_move = best_move
        else:
            tt_move = None

        if actions is None:
            actions = self.ordered_actions(
                state=state,
                actions=list(Game.actions(state)),
                maximizing=False,
                ply=ply,
                tt_move=tt_move,
            )

        best_value = math.inf
        best_move = None

        for action in actions:
            self.check_time(deadline)
            next_state = state.copy()
            Game.apply(next_state, action)
            next_hash = self.compute_hash(next_state)

            value, _ = self.max_value(
                next_state, depth - 1, alpha, beta, deadline, ply + 1, next_hash
            )

            if value < best_value:
                best_value = value
                best_move = action

            beta = min(beta, best_value)
            if beta <= alpha:
                self.add_killer_move(ply, action)
                break

        flag = self.EXACT
        if best_value <= alpha_orig:
            flag = self.UPPERBOUND
        elif best_value >= beta_orig:
            flag = self.LOWERBOUND

        self.tt_store(state_hash, depth, best_value, flag, best_move)
        return best_value, best_move

    def tt_lookup(self, state_hash, depth):
        entry = self.tt.get(state_hash)
        if entry is None:
            return None
        if entry["depth"] < depth:
            return None
        self.tt.move_to_end(state_hash)
        return entry["value"], entry["flag"], entry["best_move"]

    def tt_store(self, state_hash, depth, value, flag, best_move):
        existing = self.tt.get(state_hash)
        if existing is not None and existing["depth"] > depth:
            self.tt.move_to_end(state_hash)
            return

        self.tt[state_hash] = {
            "depth": depth,
            "value": value,
            "flag": flag,
            "best_move": best_move,
        }
        self.tt.move_to_end(state_hash)

        if len(self.tt) > self.TT_MAX_SIZE:
            self.tt.popitem(last=False)

    def tt_best_move(self, state_hash):
        entry = self.tt.get(state_hash)
        if entry is None:
            return None
        self.tt.move_to_end(state_hash)
        return entry["best_move"]

    def add_killer_move(self, ply, action):
        killers = self.killer_moves.setdefault(ply, [])
        if action in killers:
            killers.remove(action)
        killers.insert(0, action)
        if len(killers) > 2:
            killers.pop()

    def ordered_actions(self, state, actions, maximizing, ply, tt_move=None):
        killers = self.killer_moves.get(ply, [])
        scored_actions = []

        for action in actions:
            priority = 0

            if tt_move is not None and action == tt_move:
                priority += 10_000_000

            if action in killers:
                priority += 1_000_000

            next_state = state.copy()
            Game.apply(next_state, action)

            if Game.is_terminal(next_state):
                utility = Game.utility(next_state, self.player)
                if utility == 1:
                    score = 10**9
                elif utility == -1:
                    score = -10**9
                else:
                    score = 0
            else:
                score = self.evaluate(next_state)
                my_wins = self.count_immediate_wins(next_state, self.player)
                opp_wins = self.count_immediate_wins(next_state, 1 - self.player)
                score += 200_000 * my_wins
                score -= 300_000 * opp_wins

            scored_actions.append((priority + score, action))

        scored_actions.sort(key=lambda x: x[0], reverse=maximizing)
        return [action for _, action in scored_actions]

    def piece_index(self, symbol, player):
        if symbol == "x" and player == 0:
            return 0
        if symbol == "o" and player == 0:
            return 1
        if symbol == "x" and player == 1:
            return 2
        return 3

    def compute_hash(self, state):
        h = 0

        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                cell = state.board[r][c]
                if cell is not None:
                    symbol, player = cell
                    h ^= self.zobrist_board[r][c][self.piece_index(symbol, player)]

        txr, txc = state.totem_X
        tor, toc = state.totem_O
        h ^= self.zobrist_totem_x[txr][txc]
        h ^= self.zobrist_totem_o[tor][toc]

        if state.current_player == 1:
            h ^= self.zobrist_side

        h ^= self.zobrist_pieces_x[0][state.pieces_x[0]]
        h ^= self.zobrist_pieces_x[1][state.pieces_x[1]]
        h ^= self.zobrist_pieces_o[0][state.pieces_o[0]]
        h ^= self.zobrist_pieces_o[1][state.pieces_o[1]]

        return h

    def terminal_score(self, state, depth):
        utility = Game.utility(state, self.player)
        if utility > 0:
            return 1_000_000 + depth
        if utility < 0:
            return -1_000_000 - depth
        return 0

    def evaluate(self, state):
        score = 0
        board = state.board

        for r in range(6):
            for c in range(3):
                window = [board[r][c + i] for i in range(4)]
                score += self.evaluate_window(window)

        for c in range(6):
            for r in range(3):
                window = [board[r + i][c] for i in range(4)]
                score += self.evaluate_window(window)

        my_wins = self.count_immediate_wins(state, self.player)
        opp_wins = self.count_immediate_wins(state, 1 - self.player)
        score += 50_000 * my_wins
        score -= 80_000 * opp_wins

        score += 12 * self.totem_mobility(state, self.player)
        score -= 12 * self.totem_mobility(state, 1 - self.player)

        score += 8 * (state.pieces_x[self.player] - state.pieces_x[1 - self.player])
        score += 8 * (state.pieces_o[self.player] - state.pieces_o[1 - self.player])

        return score

    def evaluate_window(self, window):
        my_player = self.player
        score = 0

        my_color = 0
        opp_color = 0
        empty = 0

        my_x = 0
        my_o = 0
        opp_x = 0
        opp_o = 0

        for cell in window:
            if cell is None:
                empty += 1
                continue

            symbol, player = cell
            if player == my_player:
                my_color += 1
                if symbol == "x":
                    my_x += 1
                else:
                    my_o += 1
            else:
                opp_color += 1
                if symbol == "x":
                    opp_x += 1
                else:
                    opp_o += 1

        score += self.score_pattern(my_color, opp_color, empty, defensive=False)
        score -= self.score_pattern(opp_color, my_color, empty, defensive=True)

        score += self.score_pattern(my_x, opp_x, empty, defensive=False)
        score -= self.score_pattern(opp_x, my_x, empty, defensive=True)

        score += self.score_pattern(my_o, opp_o, empty, defensive=False)
        score -= self.score_pattern(opp_o, my_o, empty, defensive=True)

        return score

    def score_pattern(self, mine, theirs, empty, defensive):
        if mine > 0 and theirs > 0:
            return 0

        if mine == 4:
            return 200_000
        if theirs == 4:
            return 200_000

        if mine == 3 and empty == 1:
            return 4_000 if defensive else 2_500
        if mine == 2 and empty == 2:
            return 250 if defensive else 150
        if mine == 1 and empty == 3:
            return 20 if defensive else 10

        return 0

    def count_immediate_wins(self, state, player):
        original_player = state.current_player
        state.current_player = player
        try:
            count = 0
            for action in Game.actions(state):
                next_state = state.copy()
                Game.apply(next_state, action)
                if Game.is_terminal(next_state) and Game.utility(next_state, player) == 1:
                    count += 1
            return count
        finally:
            state.current_player = original_player

    def totem_mobility(self, state, player):
        original_player = state.current_player
        state.current_player = player
        try:
            return len(Game.actions(state))
        finally:
            state.current_player = original_player
