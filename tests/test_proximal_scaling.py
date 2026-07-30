import unittest

from graph_roles import find_confounder_sets


class ProximalScalingTests(unittest.TestCase):
    def test_large_common_ancestor_set_uses_verified_treatment_parents(self):
        graph = {
            "T": ["C0"],
            "Y": ["T", "C0"],
            "C0": [f"C{i}" for i in range(1, 21)],
        }
        graph.update({f"C{index}": [] for index in range(1, 21)})

        result = find_confounder_sets(graph, "T", "Y")

        self.assertEqual(result["proximal"], ["C0"])

    def test_small_graph_retains_exact_search(self):
        graph = {
            "T": ["C1", "C2"],
            "Y": ["T", "C1", "C2"],
            "C1": [],
            "C2": [],
        }

        result = find_confounder_sets(graph, "T", "Y")

        self.assertEqual(result["proximal"], ["C1", "C2"])


if __name__ == "__main__":
    unittest.main()
