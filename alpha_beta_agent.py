from agent import Agent
from oxono import Game
import math


class MyAgent(Agent):
    def __init__(self, player):
        super().__init__(player)

    def act(self, state, remaining_time):
        value, move = self.max_value(state, depth=3, alpha=-math.inf, beta=math.inf)
        return move

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
        return 0


