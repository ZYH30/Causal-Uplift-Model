import unittest

import pandas as pd

from empirical_experiments.mimic_rbc_data import (
    RBCTargetTrialSpec,
    _baseline_procedure_features,
    _baseline_service_features,
)


class MIMICRBCTemporalFeatureTests(unittest.TestCase):
    def setUp(self):
        self.cohort = pd.DataFrame(
            {
                "hadm_id": [1, 2],
                "stay_id": [11, 22],
                "intime": pd.to_datetime(
                    ["2200-01-01 00:00:00", "2200-01-02 00:00:00"]
                ),
                "trigger_hour": [8.0, 4.0],
            }
        )

    def test_latest_service_before_trigger_is_used(self):
        services = pd.DataFrame(
            {
                "hadm_id": [1, 1, 1, 2],
                "transfertime": pd.to_datetime(
                    [
                        "2200-01-01 01:00:00",
                        "2200-01-01 07:00:00",
                        "2200-01-01 09:00:00",
                        "2200-01-02 01:00:00",
                    ]
                ),
                "curr_service": ["MED", "CSURG", "NMED", "TRAUM"],
            }
        )
        result = _baseline_service_features(self.cohort, services).set_index(
            "hadm_id"
        )
        self.assertEqual(result.loc[1, "service_cardiac"], 1)
        self.assertEqual(result.loc[1, "service_neuro"], 0)
        self.assertEqual(result.loc[2, "service_trauma"], 1)

    def test_procedures_at_or_after_trigger_are_excluded(self):
        procedures = pd.DataFrame(
            {
                "hadm_id": [1, 1, 1, 2],
                "stay_id": [11, 11, 11, 22],
                "starttime": pd.to_datetime(
                    [
                        "2200-01-01 02:00:00",
                        "2200-01-01 08:00:00",
                        "2200-01-01 09:00:00",
                        "2200-01-02 01:00:00",
                    ]
                ),
                "itemid": [1, 2, 3, 4],
                "ordercategoryname": [
                    "Ventilation",
                    "Dialysis",
                    "Invasive Lines",
                    "Invasive Lines",
                ],
                "label": [
                    "Invasive Ventilation",
                    "Hemodialysis",
                    "Arterial Line",
                    "Arterial Line",
                ],
                "category": [
                    "2-Ventilation",
                    "4-Procedures",
                    "Access Lines - Invasive",
                    "Access Lines - Invasive",
                ],
            }
        )
        result = _baseline_procedure_features(
            self.cohort,
            procedures,
            RBCTargetTrialSpec(),
        ).set_index("hadm_id")
        self.assertEqual(
            result.loc[1, "pretrigger_invasive_respiratory_support"],
            1,
        )
        self.assertEqual(result.loc[1, "pretrigger_dialysis"], 0)
        self.assertEqual(result.loc[1, "pretrigger_invasive_line"], 0)
        self.assertEqual(result.loc[1, "pretrigger_procedure_count"], 1)
        self.assertEqual(result.loc[2, "pretrigger_invasive_line"], 1)


if __name__ == "__main__":
    unittest.main()
