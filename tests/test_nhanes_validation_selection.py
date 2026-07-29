import unittest

import pandas as pd

from empirical_experiments.nhanes_validation_selection import select_validation_candidate


class ValidationSelectionTests(unittest.TestCase):
    def setUp(self):
        self.rows = pd.DataFrame(
            [
                {
                    "Method": "Naive_AllFeatures",
                    "R_Loss": 10.0,
                    "DR_Qini": 0.10,
                    "PolicyOutcome_ObservedBudget": 20.0,
                },
                {
                    "Method": "OCU_CCIA_COnly",
                    "R_Loss": 8.0,
                    "DR_Qini": 0.30,
                    "PolicyOutcome_ObservedBudget": 20.1,
                },
                {
                    "Method": "OCU_CCIA_CPlusS",
                    "R_Loss": 11.0,
                    "DR_Qini": 0.05,
                    "PolicyOutcome_ObservedBudget": 19.9,
                },
            ]
        )

    def test_composite_rank_balances_all_metrics(self):
        selected = select_validation_candidate(self.rows, "composite_rank")
        self.assertEqual(selected["Method"], "OCU_CCIA_COnly")
        self.assertEqual(selected["Composite_Rank_Sum"], 5.0)

    def test_legacy_rule_is_available_for_reproduction(self):
        selected = select_validation_candidate(self.rows, "lexicographic_policy")
        self.assertEqual(selected["Method"], "OCU_CCIA_CPlusS")

    def test_unknown_rule_fails(self):
        with self.assertRaises(ValueError):
            select_validation_candidate(self.rows, "not-a-rule")


if __name__ == "__main__":
    unittest.main()
