from agent import Agent
from oxono import Game
import math


class MyAgent(Agent):
    def __init__(self, player):
        super().__init__(player)

    def act(self, state, remaining_time):
        actions = list(Game.actions(state))

        # 1. Immediate win
        for action in actions:
            next_state = state.copy()
            Game.apply(next_state, action)
            if Game.is_terminal(next_state) and Game.utility(next_state, self.player) == 1:
                return action

        # 2. Avoid moves that allow opponent immediate win
        safe_actions = []
        for action in actions:
            next_state = state.copy()
            Game.apply(next_state, action)

            opponent_can_win = False
            for opp_action in Game.actions(next_state):
                reply_state = next_state.copy()
                Game.apply(reply_state, opp_action)

                if Game.is_terminal(reply_state) and Game.utility(reply_state, self.player) == -1:
                    opponent_can_win = True
                    break

            if not opponent_can_win:
                safe_actions.append(action)

        if safe_actions:
            actions = safe_actions

        # 3. Run alpha-beta ONLY on filtered actions
        best_value = -math.inf
        best_move = None

        for action in actions:
            next_state = state.copy()
            Game.apply(next_state, action)

            value, _ = self.min_value(next_state, depth=4, alpha=-math.inf, beta=math.inf)

            if value > best_value:
                best_value = value
                best_move = action

        return best_move     

    def max_value(self, state, depth, alpha, beta):
        if Game.is_terminal(state):
            return Game.utility(state, self.player), None

        if depth == 0:
            return self.evaluate(state), None

        best_value = -math.inf
        best_move = None

        for action in Game.actions(state):
            next_state = state.copy()
            Game.apply(next_state, action)

            value, _ = self.min_value(next_state, depth - 1, alpha, beta)

            if value > best_value:
                best_value = value
                best_move = action

            alpha = max(alpha, best_value)

            if best_value >= beta:
                return best_value, best_move

        return best_value, best_move

    def min_value(self, state, depth, alpha, beta):
        if Game.is_terminal(state):
            return Game.utility(state, self.player), None

        if depth == 0:
            return self.evaluate(state), None

        best_value = math.inf
        best_move = None

        for action in Game.actions(state):
            next_state = state.copy()
            Game.apply(next_state, action)

            value, _ = self.max_value(next_state, depth - 1, alpha, beta)

            if value < best_value:
                best_value = value
                best_move = action

            beta = min(beta, best_value)

            if best_value <= alpha:
                return best_value, best_move

        return best_value, best_move

    def evaluate(self, state):
        score = 0
        board = state.board

        lines = []

        # Rows
        for r in range(6):
            lines.append([board[r][c] for c in range(6)])

        # Columns
        for c in range(6):
            lines.append([board[r][c] for r in range(6)])

        for line in lines:
            score += self.evaluate_line(line)

        return score

    def evaluate_line(self, line):
        my_player = self.player
        opp_player = 1 - self.player

        my_color_count = 0
        opp_color_count = 0
        my_x_count = 0
        my_o_count = 0
        opp_x_count = 0
        opp_o_count = 0

        for cell in line:
            if cell is None:
                continue

            symbol, player = cell

            if player == my_player:
                my_color_count += 1
                if symbol == 'x':
                    my_x_count += 1
                else:
                    my_o_count += 1
            elif player == opp_player:
                opp_color_count += 1
                if symbol == 'x':
                    opp_x_count += 1
                else:
                    opp_o_count += 1

        score = 0

        # Color-based line
        if opp_color_count == 0:
            score += self.score_count(my_color_count)
        if my_color_count == 0:
            score -= self.score_count(opp_color_count)

        # Symbol-based line: x
        if opp_x_count == 0:
            score += self.score_count(my_x_count)
        if my_x_count == 0:
            score -= self.score_count(opp_x_count)

        # Symbol-based line: o
        if opp_o_count == 0:
            score += self.score_count(my_o_count)
        if my_o_count == 0:
            score -= self.score_count(opp_o_count)

        return score

    def score_count(self, count):
        if count >= 4:
            return 10000
        if count == 3:
            return 100
        if count == 2:
            return 10
        return 0
