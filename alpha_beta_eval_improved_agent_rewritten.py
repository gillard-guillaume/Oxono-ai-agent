from agent import Agent
from oxono import Game
import math


class MyAgent(Agent):
    def __init__(self, player):
        super().__init__(player)

    def act(self, state, remaining_time):
        actions = list(Game.actions(state))
        if not actions:
            return None

        # Immediate win at root.
        for action in actions:
            next_state = state.copy()
            Game.apply(next_state, action)
            if Game.is_terminal(next_state) and Game.utility(next_state, self.player) == 1:
                return action

        # One root alpha-beta search, with ordered moves.
        value, move = self.max_value(
            state=state,
            depth=5,
            alpha=-math.inf,
            beta=math.inf,
            actions=self.ordered_actions(state, actions, maximizing=True),
        )
        return move if move is not None else actions[0]

    def max_value(self, state, depth, alpha, beta, actions=None):
        if Game.is_terminal(state):
            return self.terminal_score(state, depth), None

        if depth == 0:
            return self.evaluate(state), None

        if actions is None:
            actions = self.ordered_actions(state, list(Game.actions(state)), maximizing=True)

        best_value = -math.inf
        best_move = None

        for action in actions:
            next_state = state.copy()
            Game.apply(next_state, action)

            value, _ = self.min_value(next_state, depth - 1, alpha, beta)

            if value > best_value:
                best_value = value
                best_move = action

            alpha = max(alpha, best_value)
            if alpha >= beta:
                break

        return best_value, best_move

    def min_value(self, state, depth, alpha, beta, actions=None):
        if Game.is_terminal(state):
            return self.terminal_score(state, depth), None

        if depth == 0:
            return self.evaluate(state), None

        if actions is None:
            actions = self.ordered_actions(state, list(Game.actions(state)), maximizing=False)

        best_value = math.inf
        best_move = None

        for action in actions:
            next_state = state.copy()
            Game.apply(next_state, action)

            value, _ = self.max_value(next_state, depth - 1, alpha, beta)

            if value < best_value:
                best_value = value
                best_move = action

            beta = min(beta, best_value)
            if beta <= alpha:
                break

        return best_value, best_move

    def terminal_score(self, state, depth):
        utility = Game.utility(state, self.player)
        if utility > 0:
            return 1_000_000 + depth
        if utility < 0:
            return -1_000_000 - depth
        return 0

    def ordered_actions(self, state, actions, maximizing):
        scored_actions = []

        for action in actions:
            next_state = state.copy()
            Game.apply(next_state, action)

            # Strong priority for immediate wins.
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

                # Penalize moves that allow an immediate opponent win.
                if self.opponent_has_immediate_win(next_state):
                    score -= 500_000

            scored_actions.append((score, action))

        scored_actions.sort(key=lambda x: x[0], reverse=maximizing)
        return [action for _, action in scored_actions]

    def opponent_has_immediate_win(self, state):
        for action in Game.actions(state):
            next_state = state.copy()
            Game.apply(next_state, action)
            if Game.is_terminal(next_state) and Game.utility(next_state, self.player) == -1:
                return True
        return False

    def evaluate(self, state):
        board = state.board
        score = 0

        # Evaluate all windows of length 4 horizontally.
        for r in range(6):
            for c in range(3):
                window = [board[r][c + i] for i in range(4)]
                score += self.evaluate_window(window)

        # Evaluate all windows of length 4 vertically.
        for c in range(6):
            for r in range(3):
                window = [board[r + i][c] for i in range(4)]
                score += self.evaluate_window(window)

        # Small reserve bonus.
        score += 5 * (state.pieces_x[self.player] - state.pieces_x[1 - self.player])
        score += 5 * (state.pieces_o[self.player] - state.pieces_o[1 - self.player])

        return score

    def evaluate_window(self, window):
        my_player = self.player
        opp_player = 1 - self.player

        score = 0

        # Color counts.
        my_color = 0
        opp_color = 0
        empty = 0

        # Symbol counts.
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
                if symbol == 'x':
                    my_x += 1
                else:
                    my_o += 1
            else:
                opp_color += 1
                if symbol == 'x':
                    opp_x += 1
                else:
                    opp_o += 1

        # Color-based scoring.
        score += self.score_pattern(my_color, opp_color, empty, defensive=False)
        score -= self.score_pattern(opp_color, my_color, empty, defensive=True)

        # Symbol-based scoring for x.
        score += self.score_pattern(my_x, opp_x, empty, defensive=False)
        score -= self.score_pattern(opp_x, my_x, empty, defensive=True)

        # Symbol-based scoring for o.
        score += self.score_pattern(my_o, opp_o, empty, defensive=False)
        score -= self.score_pattern(opp_o, my_o, empty, defensive=True)

        return score

    def score_pattern(self, mine, theirs, empty, defensive):
        # Blocked pattern.
        if mine > 0 and theirs > 0:
            return 0

        # Winning / losing patterns.
        if mine == 4:
            return 100_000
        if theirs == 4:
            return 100_000

        # Open threats.
        if mine == 3 and empty == 1:
            return 800 if defensive else 500
        if mine == 2 and empty == 2:
            return 80 if defensive else 50
        if mine == 1 and empty == 3:
            return 8 if defensive else 5

        return 0
