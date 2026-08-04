import unittest

import networkx as nx

from tlcd.tlcd_runner import project_graph_to_dag


class DagProjectionTests(unittest.TestCase):
    def test_rejects_deeper_edge_that_closes_cycle(self):
        graph = {
            "Y": ["P"],
            "Collider": ["Y"],
            "P": ["Collider"],
        }
        priorities = {
            ("P", "Y"): (0.0, -0.9),
            ("Y", "Collider"): (0.0, -0.8),
            ("Collider", "P"): (1.0, -0.9),
        }

        result = project_graph_to_dag(graph, edge_priorities=priorities)
        learned = nx.DiGraph(
            (parent, child)
            for child, parents in result["graph"].items()
            for parent in parents
        )

        self.assertTrue(nx.is_directed_acyclic_graph(learned))
        self.assertTrue(learned.has_edge("P", "Y"))
        self.assertTrue(learned.has_edge("Y", "Collider"))
        self.assertFalse(learned.has_edge("Collider", "P"))
        self.assertEqual(result["rejected_edges"][0]["reason"], "would_create_cycle")

    def test_protected_query_edge_is_retained_first(self):
        graph = {
            "T": ["Y"],
            "Y": ["T"],
        }

        result = project_graph_to_dag(
            graph,
            protected_edges={("T", "Y")},
        )

        self.assertEqual(result["graph"]["Y"], ["T"])
        self.assertEqual(result["graph"]["T"], [])


if __name__ == "__main__":
    unittest.main()
