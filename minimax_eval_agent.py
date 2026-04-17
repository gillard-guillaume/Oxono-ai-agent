from agent import Agent
from oxono import Game
import math


class MyAgent(Agent):
    def __init__(self, player):
        super().__init__(player)

    def act(self, state, remaining_time):
        value, move = self.max_value(state, depth=3)
        return move

    def max_value(self, state, depth):
        if Game.is_terminal(state):
            return Game.utility(state, self.player), None

        if depth == 0:
            return self.evaluate(state), None

        best_value = -math.inf
        best_move = None

        for action in Game.actions(state):
            next_state = state.copy()
            Game.apply(next_state, action)

            value, _ = self.min_value(next_state, depth - 1)

            if value > best_value:
                best_value = value
                best_move = action

        return best_value, best_move

    def min_value(self, state, depth):
        if Game.is_terminal(state):
            return Game.utility(state, self.player), None

        if depth == 0:
            return self.evaluate(state), None

        best_value = math.inf
        best_move = None

        for action in Game.actions(state):
            next_state = state.copy()
            Game.apply(next_state, action)

            value, _ = self.max_value(next_state, depth - 1)

            if value < best_value:
                best_value = value
                best_move = action

        return best_value, best_move


    def evaluate(self, state):
        score = 0
        board = state.board
        opponent = 1 - self.player

        lines = []

        # Rows
        for r in range(6):
            lines.append([board[r][c] for c in range(6)])

        # Columns
        for c in range(6):
            lines.append([board[r][c] for r in range(6)])

        for line in lines:
            my_color = 0
            opp_color = 0
            my_x = 0
            my_o = 0
            opp_x = 0
            opp_o = 0

            for cell in line:
                if cell is None:
                    continue

                symbol, player = cell
                if player == self.player:
                    my_color += 1
                    if symbol == 'x':
                        my_x += 1
                    else:
                        my_o += 1
                elif player == opponent:
                    opp_color += 1
                    if symbol == 'x':
                        opp_x += 1
                    else:
                        opp_o += 1

            # Color alignments
            if opp_color == 0:
                score += self.score_count(my_color)
            if my_color == 0:
                score -= self.score_count(opp_color)

            # Symbol alignments
            if opp_x == 0:
                score += self.score_count(my_x)
            if my_x == 0:
                score -= self.score_count(opp_x)

            if opp_o == 0:
                score += self.score_count(my_o)
            if my_o == 0:
                score -= self.score_count(opp_o)

        return score

    def score_count(self, count):
        if count >= 4:
            return 10000
        if count == 3:
            return 100
        if count == 2:
            return 10
        return 0
