import unittest
from agents.ab7_agent import AB7
from oxono.oxono import Game, State


class TestZobristHash(unittest.TestCase):

    def setUp(self):
        self.agent = AB7(player=0)

    def test_hash_consistency_after_one_move(self):
        state = State()
        h = self.agent.hash_state(state)

        for action in Game.actions(state):
            new_state = state.copy()
            Game.apply(new_state, action)

            h_incremental = self.agent.update_hash(h, state, action)
            h_full = self.agent.hash_state(new_state)

            self.assertEqual(h_incremental, h_full)

    def test_hash_consistency_multiple_moves(self):
        state = State()
        h = self.agent.hash_state(state)

        for _ in range(10):
            actions = Game.actions(state)
            if not actions:
                break

            action = actions[0]  # ou random.choice(actions)

            new_state = state.copy()
            Game.apply(new_state, action)

            h = self.agent.update_hash(h, state, action)
            h_full = self.agent.hash_state(new_state)

            self.assertEqual(h, h_full)

            state = new_state


if __name__ == "__main__":
    unittest.main()