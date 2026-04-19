import unittest
from agents.ab7_agent import AB7, EXACT, LOWERBOUND, UPPERBOUND
from collections import OrderedDict


class TestTranspositionTable(unittest.TestCase):
    def setUp(self):
        self.agent = AB7(player=0)
        self.agent.tt = OrderedDict()
        self.agent.tt_size = 3  # petit pour tester LRU

    # ----------------------
    # BASIC LOOKUP TESTS
    # ----------------------

    def test_tt_lookup_exact(self):
        h = 12345
        val = 10.0
        depth = 5
        alpha = -float('inf')
        beta = float('inf')

        self.agent.tt[h] = (val, depth, None, EXACT)

        result_val, result_move, new_alpha, new_beta = self.agent.tt_lookup(h, depth, alpha, beta)

        self.assertEqual(result_val, val)
        self.assertEqual(new_alpha, alpha)
        self.assertEqual(new_beta, beta)

    def test_tt_lookup_lowerbound_cutoff(self):
        h = 123
        self.agent.tt[h] = (15.0, 5, None, LOWERBOUND)

        val, _, alpha, beta = self.agent.tt_lookup(h, 5, -float('inf'), 10.0)

        self.assertEqual(val, 15.0)

    def test_tt_lookup_upperbound_cutoff(self):
        h = 123
        self.agent.tt[h] = (5.0, 5, None, UPPERBOUND)

        val, _, alpha, beta = self.agent.tt_lookup(h, 5, 10.0, float('inf'))

        self.assertEqual(val, 5.0)

    def test_tt_lookup_no_cutoff_updates_alpha(self):
        h = 123
        self.agent.tt[h] = (7.0, 5, None, LOWERBOUND)

        val, _, alpha, beta = self.agent.tt_lookup(h, 5, -float('inf'), 10.0)

        self.assertIsNone(val)
        self.assertEqual(alpha, 7.0)

    def test_tt_lookup_no_cutoff_updates_beta(self):
        h = 123
        self.agent.tt[h] = (7.0, 5, None, UPPERBOUND)

        val, _, alpha, beta = self.agent.tt_lookup(h, 5, 0.0, float('inf'))

        self.assertIsNone(val)
        self.assertEqual(beta, 7.0)

    # ----------------------
    # DEPTH HANDLING
    # ----------------------

    def test_tt_lookup_depth_too_shallow(self):
        h = 1
        self.agent.tt[h] = (10.0, 3, None, EXACT)

        val, _, _, _ = self.agent.tt_lookup(h, 5, -float('inf'), float('inf'))

        self.assertIsNone(val)

    def test_tt_lookup_depth_sufficient(self):
        h = 1
        self.agent.tt[h] = (10.0, 5, None, EXACT)

        val, _, _, _ = self.agent.tt_lookup(h, 5, -float('inf'), float('inf'))

        self.assertEqual(val, 10.0)

    # ----------------------
    # STORE LOGIC
    # ----------------------

    def test_tt_store_replaces_if_deeper(self):
        h = 1
        self.agent.tt[h] = (5.0, 3, None, EXACT)

        self.agent.tt_store(h, 10.0, 5, None, EXACT)

        val, depth, _, _ = self.agent.tt[h]
        self.assertEqual(val, 10.0)
        self.assertEqual(depth, 5)

    def test_tt_store_does_not_replace_if_shallower(self):
        h = 1
        self.agent.tt[h] = (10.0, 5, None, EXACT)

        self.agent.tt_store(h, 3.0, 3, None, EXACT)

        val, depth, _, _ = self.agent.tt[h]
        self.assertEqual(val, 10.0)
        self.assertEqual(depth, 5)

    def test_tt_store_exact_not_overwritten_by_bound_same_depth(self):
        h = 1
        self.agent.tt[h] = (10.0, 5, None, EXACT)

        self.agent.tt_store(h, 5.0, 5, None, LOWERBOUND)

        val, depth, _, flag = self.agent.tt[h]
        self.assertEqual(flag, EXACT)

    # ----------------------
    # LRU BEHAVIOR
    # ----------------------

    def test_tt_lru_eviction(self):
        self.agent.tt_store(1, 1, 1, None, EXACT)
        self.agent.tt_store(2, 2, 1, None, EXACT)
        self.agent.tt_store(3, 3, 1, None, EXACT)

        # access key 1 to make it recent
        self.agent.tt_lookup(1, 1, -float('inf'), float('inf'))

        # add new entry -> should evict oldest (key 2)
        self.agent.tt_store(4, 4, 1, None, EXACT)

        self.assertNotIn(2, self.agent.tt)
        self.assertIn(1, self.agent.tt)
        self.assertIn(3, self.agent.tt)
        self.assertIn(4, self.agent.tt)

    # ----------------------
    # MOVE HANDLING
    # ----------------------

    def test_tt_lookup_returns_move_even_on_miss(self):
        h = 1
        move = ("O", (0, 0), (1, 1))
        self.agent.tt[h] = (10.0, 3, move, EXACT)

        val, returned_move, _, _ = self.agent.tt_lookup(h, 5, -float('inf'), float('inf'))

        self.assertIsNone(val)
        self.assertEqual(returned_move, move)


if __name__ == '__main__':
    unittest.main()