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
        return 0
