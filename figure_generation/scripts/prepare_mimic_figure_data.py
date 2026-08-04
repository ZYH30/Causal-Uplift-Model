"""Regenerate frozen MIMIC figure inputs from aggregate public artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DEFAULT_RESULTS = PROJECT_ROOT / "results" / "mimic_rbc_chart_augmented"
FIGURE_DATA = HERE.parent / "data"


def prepare_figure_data(results: Path) -> None:
    final = results / "final"
    FIGURE_DATA.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(final / "common_support_cohort_audit.csv")
    common_support = cohort.loc[cohort["population"].eq("common_support")].copy()
    era_labels = {
        "discovery_2008_2016": ("2008--2016", "Discovery"),
        "validation_2017_2019": ("2017--2019", "Selection"),
        "temporal_2020_2022": ("2020--2022", "Temporal test"),
    }
    eras = common_support.loc[common_support["era"].isin(era_labels)].copy()
    eras["calendar_era"] = eras["era"].map(lambda value: era_labels[value][0])
    eras["analysis_role"] = eras["era"].map(lambda value: era_labels[value][1])
    eras[
        [
            "calendar_era",
            "analysis_role",
            "n",
            "treated",
            "treatment_rate",
            "outcomes",
            "outcome_rate",
        ]
    ].to_csv(FIGURE_DATA / "mimic_rbc_analysis_eras.csv", index=False)

    original = cohort.loc[
        cohort["era"].eq("overall") & cohort["population"].eq("original")
    ].iloc[0]
    analysis = cohort.loc[
        cohort["era"].eq("overall")
        & cohort["population"].eq("common_support")
    ].iloc[0]
    pd.DataFrame(
        {
            "stage": [
                "qualifying_triggers",
                "excluded_prior_rbc",
                "excluded_unavailable_landmark",
                "target_trial_eligible",
                "excluded_no_support",
                "analysis_cohort",
                "early_rbc",
                "deaths_90d",
            ],
            "count": [
                18_742,
                2_432,
                742,
                int(original["n"]),
                int(original["n"] - analysis["n"]),
                int(analysis["n"]),
                int(analysis["treated"]),
                int(analysis["outcomes"]),
            ],
        }
    ).to_csv(FIGURE_DATA / "mimic_rbc_cohort_flow.csv", index=False)

    balance = pd.read_csv(final / "temporal_balance.csv")
    balance["abs_smd_pre"] = balance["smd_pre"].abs()
    balance["abs_smd_weighted"] = balance["smd_ipw"].abs()
    balance.to_csv(FIGURE_DATA / "mimic_rbc_temporal_balance.csv", index=False)

    diagnostics = json.loads((final / "temporal_diagnostics.json").read_text())
    pd.DataFrame([diagnostics]).to_csv(
        FIGURE_DATA / "mimic_rbc_temporal_diagnostics.csv", index=False
    )
    pd.read_csv(final / "post_discovery_repeated_summary.csv").to_csv(
        FIGURE_DATA / "mimic_rbc_repeated_summary.csv", index=False
    )

    bootstrap = pd.read_csv(final / "temporal_paired_bootstrap_summary.csv")
    bootstrap.loc[
        bootstrap["intervention"].str.endswith("_CPlusS"), "intervention"
    ] = "RITE_CPlusS"
    bootstrap.loc[
        bootstrap["comparison"].eq("TLCD_chart_vs_all_chart")
    ].to_csv(FIGURE_DATA / "mimic_rbc_temporal_comparison.csv", index=False)

    copies = {
        "temporal_benefit_profiles.csv": "mimic_rbc_temporal_benefit_profiles.csv",
        "temporal_hb_mechanism.csv": "mimic_rbc_temporal_hb_mechanism.csv",
        "temporal_chart_profiles.csv": "mimic_rbc_temporal_chart_profiles.csv",
        "temporal_chart_feature_gradients.csv": (
            "mimic_rbc_temporal_chart_feature_gradients.csv"
        ),
    }
    for source, destination in copies.items():
        pd.read_csv(final / source).to_csv(FIGURE_DATA / destination, index=False)

    policy = pd.read_csv(final / "temporal_policy_budget_curve.csv")
    policy.loc[policy["Interface"].str.endswith("_CPlusS"), "Interface"] = (
        "RITE_CPlusS"
    )
    policy.to_csv(
        FIGURE_DATA / "mimic_rbc_temporal_policy_budget_curve.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    prepare_figure_data(args.results_dir)


if __name__ == "__main__":
    main()
