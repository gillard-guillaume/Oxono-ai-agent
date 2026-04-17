from agent import Agent
from oxono import Game
import math
import random
import time


class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action

        self.children = []
        self.visits = 0
        self.value = 0

        self.untried_actions = list(Game.actions(state))


class MCTSAgent(Agent):
    def __init__(self, player):
        super().__init__(player)

    def act(self, state, remaining_time):
        root = Node(state.copy())

        end_time = time.time() + min(1.0, remaining_time)  # 1 sec per move baseline

        while time.time() < end_time:
            node = root

            # SELECT
            while node.untried_actions == [] and node.children:
                node = self.uct_select(node)

            # EXPAND
            if node.untried_actions:
                node = self.expand(node)

            # SIMULATE
            result_state = self.simulate(node.state)

            # BACKPROPAGATE
            self.backpropagate(node, result_state)

        # Choose most visited move
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action

    def uct_select(self, node, c=1.4):
        return max(
            node.children,
            key=lambda child: (child.value / child.visits) +
            c * math.sqrt(math.log(node.visits) / child.visits)
        )

    def expand(self, node):
        action = node.untried_actions.pop()
        next_state = node.state.copy()
        Game.apply(next_state, action)

        child = Node(next_state, parent=node, action=action)
        node.children.append(child)
        return child

    def simulate(self, state):
        sim_state = state.copy()

        while not Game.is_terminal(sim_state):
            actions = list(Game.actions(sim_state))
            action = random.choice(actions)
            Game.apply(sim_state, action)

        return sim_state

    def backpropagate(self, node, result_state):
        while node is not None:
            node.visits += 1
            node.value += Game.utility(result_state, self.player)
            node = node.parent
