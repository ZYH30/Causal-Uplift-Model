"""Extract time-safe MIMIC-IV chartevents aggregates from BigQuery."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from google.cloud import bigquery


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_DATABASE = PROJECT_ROOT / "data" / "mimic_iv_tlcd.sqlite"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "generated" / "mimic-rbc-chart"
DEFAULT_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
DEFAULT_DATASET = "tlcd_chartevents_tmp"

ITEMS = {
    "heart_rate": [220045],
    "sbp": [220050, 220179],
    "dbp": [220051, 220180],
    "map": [220052, 220181],
    "resp_rate": [220210],
    "spo2": [220277],
    "temp_c": [223761, 223762],
    "cvp": [220074],
    "fio2": [223835],
    "o2_flow": [223834, 227287, 227582],
    "gcs_eye": [220739],
    "gcs_verbal": [223900],
    "gcs_motor": [223901],
    "gcs_total": [226755, 227013],
    "rass": [228096],
    "weight_kg": [224639, 226512],
    "peep": [220339],
    "tidal_volume": [224685, 224686],
    "peak_pressure": [224695],
}

RANGES = {
    "heart_rate": (20.0, 250.0),
    "sbp": (30.0, 300.0),
    "dbp": (10.0, 200.0),
    "map": (20.0, 200.0),
    "resp_rate": (2.0, 80.0),
    "spo2": (50.0, 100.0),
    "temp_c": (30.0, 43.0),
    "cvp": (-20.0, 50.0),
    "fio2": (21.0, 100.0),
    "o2_flow": (0.0, 100.0),
    "gcs_eye": (1.0, 4.0),
    "gcs_verbal": (1.0, 5.0),
    "gcs_motor": (1.0, 6.0),
    "gcs_total": (3.0, 15.0),
    "rass": (-5.0, 4.0),
    "weight_kg": (20.0, 400.0),
    "peep": (0.0, 40.0),
    "tidal_volume": (50.0, 2000.0),
    "peak_pressure": (0.0, 80.0),
}


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def cohort_trigger_table(database: Path) -> pd.DataFrame:
    """Reconstruct the published cohort keys without reading chartevents."""
    from empirical_experiments.mimic_rbc_data import (
        build_rbc_landmark_cohort,
    )
    from empirical_experiments.mimic_screen_candidates import clean_labs

    connection = sqlite3.connect(database)
    try:
        stays = pd.read_sql_query(
            """
            SELECT subject_id, hadm_id, stay_id, gender, age, anchor_year_group,
                   first_careunit, admission_type, admission_location, insurance,
                   race, admittime, intime, outtime, icu_hours, death_time,
                   days_to_death_from_icu
            FROM stays
            """,
            connection,
        )
        labs = pd.read_sql_query(
            """
            SELECT hadm_id, hour_from_icu, feature, valuenum
            FROM labevents_selected
            """,
            connection,
        )
        inputs = pd.read_sql_query(
            """
            SELECT hadm_id, stay_id, start_hour, treatment_group, amount
            FROM inputevents_selected
            """,
            connection,
        )
    finally:
        connection.close()

    for column in ("admittime", "intime", "outtime"):
        stays[column] = pd.to_datetime(stays[column], errors="coerce")
    cohort, flow = build_rbc_landmark_cohort(stays, clean_labs(labs), inputs)
    if flow["landmark_eligible"] != 15_568:
        raise ValueError(f"Cohort drift detected: {flow}")
    trigger = cohort[
        ["subject_id", "hadm_id", "stay_id", "intime", "trigger_hour", "T", "Y"]
    ].copy()
    trigger["trigger_time"] = trigger["intime"] + pd.to_timedelta(
        trigger["trigger_hour"], unit="h"
    )
    if trigger["trigger_time"].isna().any():
        raise ValueError("Missing trigger_time in reconstructed cohort")
    return trigger


def _feature_case() -> str:
    clauses = []
    for feature, itemids in ITEMS.items():
        ids = ", ".join(str(itemid) for itemid in itemids)
        clauses.append(f"WHEN ce.itemid IN ({ids}) THEN '{feature}'")
    return "CASE\n      " + "\n      ".join(clauses) + "\n    END"


def _value_case() -> str:
    text_ids = set(ITEMS["gcs_eye"] + ITEMS["gcs_verbal"] + ITEMS["gcs_motor"])
    text_ids.update(ITEMS["rass"])
    text_ids_string = ", ".join(str(itemid) for itemid in sorted(text_ids))
    fahrenheit = ", ".join(str(itemid) for itemid in [223761])
    fio2 = ", ".join(str(itemid) for itemid in ITEMS["fio2"])
    return f"""
    CASE
      WHEN ce.itemid IN ({fahrenheit}) THEN (ce.valuenum - 32.0) * 5.0 / 9.0
      WHEN ce.itemid IN ({fio2}) AND ce.valuenum <= 1.5 THEN ce.valuenum * 100.0
      WHEN ce.itemid IN ({text_ids_string}) THEN COALESCE(
        ce.valuenum,
        SAFE_CAST(REGEXP_EXTRACT(ce.value, r'-?[0-9]+') AS FLOAT64)
      )
      ELSE ce.valuenum
    END
    """.strip()


def _range_predicate() -> str:
    clauses = [
        f"(feature = '{feature}' AND numeric_value BETWEEN {low} AND {high})"
        for feature, (low, high) in RANGES.items()
    ]
    return "\n      OR ".join(clauses)


def aggregate_query(cohort_table: str) -> str:
    itemids = sorted({itemid for values in ITEMS.values() for itemid in values})
    itemid_sql = ", ".join(str(itemid) for itemid in itemids)
    return f"""
WITH candidate_events AS (
  SELECT
    c.stay_id,
    ce.charttime,
    {_feature_case()} AS feature,
    {_value_case()} AS numeric_value
  FROM `physionet-data.mimiciv_3_1_icu.chartevents` AS ce
  JOIN `{cohort_table}` AS c
    USING (stay_id)
  WHERE ce.itemid IN ({itemid_sql})
    AND ce.charttime >= DATETIME_SUB(c.trigger_time, INTERVAL 24 HOUR)
    AND ce.charttime < c.trigger_time
), valid_events AS (
  SELECT *
  FROM candidate_events
  WHERE numeric_value IS NOT NULL
    AND (
      {_range_predicate()}
    )
), aggregate AS (
  SELECT
    stay_id,
    feature,
    ARRAY_AGG(numeric_value ORDER BY charttime DESC LIMIT 1)[OFFSET(0)] AS last,
    ARRAY_AGG(numeric_value ORDER BY charttime ASC LIMIT 1)[OFFSET(0)] AS first,
    MIN(numeric_value) AS min,
    MAX(numeric_value) AS max,
    AVG(numeric_value) AS mean,
    STDDEV_POP(numeric_value) AS sd,
    SAFE_DIVIDE(
      COVAR_POP(
        numeric_value,
        DATETIME_DIFF(charttime, DATETIME '2000-01-01 00:00:00', SECOND)
      ),
      VAR_POP(DATETIME_DIFF(charttime, DATETIME '2000-01-01 00:00:00', SECOND))
    ) * 3600.0 AS slope_per_hour,
    COUNT(*) AS count,
    COUNTIF(feature = 'heart_rate' AND numeric_value > 100) / COUNT(*) AS high_fraction,
    COUNTIF(
      (feature = 'map' AND numeric_value < 65)
      OR (feature = 'sbp' AND numeric_value < 90)
      OR (feature = 'spo2' AND numeric_value < 92)
    ) / COUNT(*) AS low_fraction
  FROM valid_events
  GROUP BY stay_id, feature
)
SELECT *
FROM aggregate
ORDER BY stay_id, feature
""".strip()


def wide_chart_features(aggregate: pd.DataFrame) -> pd.DataFrame:
    statistics = [
        "last",
        "first",
        "min",
        "max",
        "mean",
        "sd",
        "slope_per_hour",
        "count",
        "high_fraction",
        "low_fraction",
    ]
    blocks = []
    for statistic in statistics:
        pivot = aggregate.pivot(index="stay_id", columns="feature", values=statistic)
        pivot.columns = [f"chart_{feature}_{statistic}" for feature in pivot.columns]
        blocks.append(pivot)
    wide = pd.concat(blocks, axis=1).reset_index()

    def column(name: str) -> pd.Series:
        if name in wide:
            return pd.to_numeric(wide[name], errors="coerce")
        return pd.Series(np.nan, index=wide.index, dtype=float)

    component_total = (
        column("chart_gcs_eye_last")
        + column("chart_gcs_verbal_last")
        + column("chart_gcs_motor_last")
    )
    direct_total = column("chart_gcs_total_last")
    wide["chart_gcs_combined_last"] = direct_total.combine_first(component_total)
    wide["chart_shock_index_last"] = column("chart_heart_rate_last") / column(
        "chart_sbp_last"
    ).replace(0, np.nan)
    wide["chart_pulse_pressure_last"] = column("chart_sbp_last") - column(
        "chart_dbp_last"
    )
    wide["chart_map_deficit"] = (65.0 - column("chart_map_min")).clip(lower=0)
    count_columns = [
        name
        for name in wide.columns
        if name.startswith("chart_") and name.endswith("_count")
    ]
    wide["chart_measurement_count"] = wide[count_columns].sum(axis=1, min_count=1)
    core_last = [
        "chart_heart_rate_last",
        "chart_sbp_last",
        "chart_map_last",
        "chart_resp_rate_last",
        "chart_spo2_last",
        "chart_temp_c_last",
        "chart_gcs_combined_last",
    ]
    wide["chart_core_vitals_observed"] = wide[
        [name for name in core_last if name in wide]
    ].notna().sum(axis=1)
    return wide


def upload_cohort(
    client: bigquery.Client,
    trigger: pd.DataFrame,
    dataset_name: str,
) -> str:
    dataset_id = f"{client.project}.{dataset_name}"
    dataset = bigquery.Dataset(dataset_id)
    dataset.location = "US"
    dataset.default_table_expiration_ms = 24 * 60 * 60 * 1000
    client.create_dataset(dataset, exists_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    table_id = f"{dataset_id}.rbc_cohort_{timestamp}"
    rows = trigger[["stay_id", "trigger_time"]].copy()
    rows["stay_id"] = rows["stay_id"].astype("int64")
    rows["trigger_time"] = pd.to_datetime(rows["trigger_time"]).dt.strftime(
        "%Y-%m-%d %H:%M:%S.%f"
    )
    schema = [
        bigquery.SchemaField("stay_id", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("trigger_time", "DATETIME", mode="REQUIRED"),
    ]
    job_config = bigquery.LoadJobConfig(
        schema=schema,
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    csv_bytes = rows.to_csv(index=False).encode("utf-8")
    import io

    job = client.load_table_from_file(
        io.BytesIO(csv_bytes), table_id, job_config=job_config, rewind=True
    )
    job.result()
    loaded = client.get_table(table_id)
    if loaded.num_rows != len(rows):
        raise ValueError(f"Uploaded {loaded.num_rows} rows, expected {len(rows)}")
    return table_id


def extract(args: argparse.Namespace) -> None:
    if not args.project:
        raise ValueError(
            "Set GOOGLE_CLOUD_PROJECT or pass --project with a Google Cloud "
            "project authorized for MIMIC-IV BigQuery access."
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.out_dir / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    trigger = cohort_trigger_table(args.database)
    trigger.to_pickle(private_dir / "mimic_rbc_trigger_cohort.pkl.gz")

    client = bigquery.Client(project=args.project)
    table_id = upload_cohort(client, trigger, args.dataset)
    query = aggregate_query(table_id)
    (args.out_dir / "chartevents_aggregate_query.sql").write_text(
        query, encoding="utf-8"
    )
    dry_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    dry_job = client.query(query, job_config=dry_config)
    started = datetime.now(timezone.utc)
    try:
        query_job = client.query(query)
        aggregate = query_job.result().to_dataframe(create_bqstorage_client=False)
    finally:
        if not args.keep_temp_table:
            client.delete_table(table_id, not_found_ok=True)

    aggregate.to_pickle(private_dir / "chartevents_aggregate_long.pkl.gz")
    wide = wide_chart_features(aggregate)
    wide.to_pickle(private_dir / "chartevents_features_wide.pkl.gz")
    manifest = {
        "source_table": "physionet-data.mimiciv_3_1_icu.chartevents",
        "billing_project": args.project,
        "cohort_n": int(len(trigger)),
        "cohort_treated": int(trigger["T"].sum()),
        "cohort_outcomes": int(trigger["Y"].sum()),
        "window": "[trigger_time - 24 hours, trigger_time)",
        "itemids": ITEMS,
        "plausible_ranges": RANGES,
        "estimated_bytes_processed": int(dry_job.total_bytes_processed),
        "actual_bytes_processed": int(query_job.total_bytes_processed),
        "aggregate_rows": int(len(aggregate)),
        "stays_with_chart_features": int(wide["stay_id"].nunique()),
        "wide_feature_count": int(len(wide.columns) - 1),
        "temporary_table_deleted": not args.keep_temp_table,
        "started_utc": started.isoformat(),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "patient_level_raw_events_exported": False,
    }
    write_json(manifest, args.out_dir / "chartevents_extraction_manifest.json")
    print(json.dumps(manifest, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--keep-temp-table", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    extract(parse_args())
