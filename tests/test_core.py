import math
import unittest
from llmrec.data import build_dpo_pairs, select_trajectories
from llmrec.objectives import build_sequence, causal_cross_entropy, dpo_loss
from llmrec.evaluation import evaluate

class CoreTests(unittest.TestCase):
    def test_mask_preserves_input(self):
        original = build_sequence([0], [1, 2], [3, 4])
        masked = build_sequence([0], [1, 2], [3, 4], True)
        self.assertEqual(original['input_ids'], masked['input_ids'])
        self.assertEqual(masked['labels'], [-100, 1, 2, -100, -100])
        self.assertEqual(original['labels'], [-100, 1, 2, 3, 4])

    def test_causal_shift(self):
        self.assertLess(causal_cross_entropy([[0, 12, 0], [0, 0, 12], [12, 0, 0]], [-100, 1, 2]), 0.001)
        with self.assertRaises(ValueError):
            causal_cross_entropy([[0, 0], [0, 0]], [-100, -100])

    def test_negative_exclusions(self):
        chosen = 'prod:1:2:3'
        pairs = build_dpo_pairs(chosen, [chosen, 'prod:1:4:5', 'prod:6:7:8', 'video:1:2:3'], history=['prod:6:7:8'])
        self.assertEqual([p['rejected'] for p in pairs], ['prod:1:4:5'])

    def test_dpo_reference(self):
        self.assertAlmostEqual(dpo_loss(-1, -2, -1, -2), math.log(2))
        self.assertLess(dpo_loss(-0.5, -2, -1, -2), math.log(2))

    def test_invalid_outputs_keep_request_and_budget(self):
        catalog = ['prod:1:2:3']
        rows = [{'domain':'prod', 'targets':catalog, 'think':['invalid', catalog[0]], 'nothink':[]}]
        result = evaluate(rows, catalog, k=1)
        self.assertEqual(result['sid_hit_rate'], 0)
        self.assertEqual(result['n_requests'], 1)

    def test_trajectory_dedup(self):
        rows = [{'reasoning':'same', 'sid':'prod:1:2:3'}]*3
        self.assertEqual(len(select_trajectories(rows, 'prod:1:2:3')), 1)

if __name__ == '__main__':
    unittest.main()
