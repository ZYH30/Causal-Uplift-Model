import argparse
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MIMIC_ROOT = (
    PROJECT_ROOT.parent
    / "MIMIC-IV"
    / "physionet.org"
    / "files"
    / "mimiciv"
    / "3.1"
)

INPUT_ITEM_GROUPS = {
    "rbc": {220996, 225168, 226368, 227070},
    "vasopressor": {
        221289,
        221653,
        221662,
        221749,
        221906,
        222315,
        229617,
        229630,
        229631,
        229632,
    },
    "insulin": {223257, 223258, 223259, 223260, 223261, 223262, 229299, 229619},
    "furosemide": {221794, 228340},
    "crystalloid": {220954, 220955, 225158, 225159, 225828},
}

CHART_ITEM_NAMES = {
    220045: "heart_rate",
    220050: "arterial_sbp",
    220051: "arterial_dbp",
    220052: "arterial_map",
    220179: "noninvasive_sbp",
    220180: "noninvasive_dbp",
    220181: "noninvasive_map",
    220210: "respiratory_rate",
    220277: "spo2",
    223761: "temperature_f",
    223762: "temperature_c",
    224639: "daily_weight",
    226512: "admission_weight",
}

LAB_ITEM_NAMES = {
    50811: "hemoglobin_blood_gas",
    50813: "lactate_blood_gas",
    50862: "albumin",
    50882: "bicarbonate",
    50885: "bilirubin_total",
    50912: "creatinine",
    50931: "glucose",
    50960: "magnesium",
    50971: "potassium",
    50983: "sodium",
    51006: "bun",
    51221: "hematocrit",
    51222: "hemoglobin",
    51237: "inr",
    51244: "lymphocytes",
    51250: "mcv",
    51265: "platelets",
    51274: "pt",
    51275: "ptt",
    51301: "wbc",
    51640: "hemoglobin_chemistry",
    52442: "lactate_blood_gas_new",
    52546: "creatinine_new",
    52569: "glucose_new",
    52610: "potassium_new",
    52623: "sodium_new",
    52647: "bun_new",
    52923: "ptt_new",
    53085: "albumin_new",
    53154: "lactate",
    53189: "platelets_new",
}

OUTPUT_ITEM_GROUPS = {
    "urine": {226559, 226560, 226561, 226563, 226567, 226584},
}


def connect_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA cache_size=-262144")
    return connection


def table_exists(connection: sqlite3.Connection, table: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def source_path(root: Path, module: str, filename: str) -> Path:
    path = root / module / filename
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def read_base_tables(root: Path) -> pd.DataFrame:
    patients = pd.read_csv(
        source_path(root, "hosp", "patients.csv.gz"),
        usecols=["subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group", "dod"],
    )
    admissions = pd.read_csv(
        source_path(root, "hosp", "admissions.csv.gz"),
        usecols=[
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "deathtime",
            "admission_type",
            "admission_location",
            "insurance",
            "race",
            "hospital_expire_flag",
        ],
    )
    stays = pd.read_csv(source_path(root, "icu", "icustays.csv.gz"))
    stays["intime"] = pd.to_datetime(stays["intime"], errors="coerce")
    stays["outtime"] = pd.to_datetime(stays["outtime"], errors="coerce")
    stays = stays.sort_values(["subject_id", "intime", "stay_id"])
    stays["icu_stay_number"] = stays.groupby("subject_id").cumcount() + 1
    stays = stays[stays["icu_stay_number"] == 1].copy()
    stays = stays.merge(patients, on="subject_id", how="inner", validate="many_to_one")
    stays = stays.merge(
        admissions,
        on=["subject_id", "hadm_id"],
        how="inner",
        validate="many_to_one",
    )
    stays["age"] = (
        stays["anchor_age"]
        + stays["intime"].dt.year
        - stays["anchor_year"]
    )
    stays["age"] = stays["age"].clip(lower=0, upper=100)
    stays = stays[(stays["age"] >= 18) & stays["intime"].notna()].copy()
    stays["icu_hours"] = (
        (stays["outtime"] - stays["intime"]).dt.total_seconds() / 3600.0
    )
    stays["admittime"] = pd.to_datetime(stays["admittime"], errors="coerce")
    stays["dischtime"] = pd.to_datetime(stays["dischtime"], errors="coerce")
    stays["deathtime"] = pd.to_datetime(stays["deathtime"], errors="coerce")
    stays["dod"] = pd.to_datetime(stays["dod"], errors="coerce")
    stays["death_time"] = stays[["deathtime", "dod"]].min(axis=1)
    stays["days_to_death_from_icu"] = (
        (stays["death_time"] - stays["intime"]).dt.total_seconds() / 86400.0
    )
    stays["mortality_28d"] = (
        stays["days_to_death_from_icu"].between(0, 28, inclusive="both")
    ).astype(int)
    stays["mortality_90d"] = (
        stays["days_to_death_from_icu"].between(0, 90, inclusive="both")
    ).astype(int)
    stays["mortality_hospital"] = stays["hospital_expire_flag"].fillna(0).astype(int)
    return stays.reset_index(drop=True)


def write_base_tables(
    connection: sqlite3.Connection,
    root: Path,
    rebuild: bool,
) -> pd.DataFrame:
    if table_exists(connection, "stays") and not rebuild:
        stays = pd.read_sql_query("SELECT * FROM stays", connection)
        stays["intime"] = pd.to_datetime(stays["intime"], errors="coerce")
        stays["outtime"] = pd.to_datetime(stays["outtime"], errors="coerce")
        return stays

    stays = read_base_tables(root)
    connection.execute("DROP TABLE IF EXISTS stays")
    stays.to_sql("stays", connection, if_exists="replace", index=False)
    connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_stays_stay ON stays(stay_id)")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_stays_hadm ON stays(hadm_id)")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_stays_subject ON stays(subject_id)")

    diagnoses = pd.read_csv(
        source_path(root, "hosp", "diagnoses_icd.csv.gz"),
        usecols=["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
    )
    diagnoses = diagnoses[diagnoses["hadm_id"].isin(set(stays["hadm_id"]))].copy()
    connection.execute("DROP TABLE IF EXISTS diagnoses")
    diagnoses.to_sql("diagnoses", connection, if_exists="replace", index=False)
    connection.execute("CREATE INDEX IF NOT EXISTS idx_diag_hadm ON diagnoses(hadm_id)")
    connection.commit()
    return stays


def time_maps(stays: pd.DataFrame) -> tuple[dict[int, pd.Timestamp], dict[int, pd.Timestamp]]:
    stay_map = dict(zip(stays["stay_id"].astype(int), pd.to_datetime(stays["intime"])))
    hadm_map = dict(zip(stays["hadm_id"].astype(int), pd.to_datetime(stays["intime"])))
    return stay_map, hadm_map


def append_frame(
    connection: sqlite3.Connection,
    table: str,
    frame: pd.DataFrame,
    first_write: bool,
) -> bool:
    if frame.empty:
        return first_write
    frame.to_sql(
        table,
        connection,
        if_exists="replace" if first_write else "append",
        index=False,
    )
    return False


def stream_chartevents(
    connection: sqlite3.Connection,
    root: Path,
    stays: pd.DataFrame,
    chunksize: int,
    rebuild: bool,
    min_hour: float,
    max_hour: float,
) -> dict:
    table = "chartevents_selected"
    if table_exists(connection, table) and not rebuild:
        count = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return {"rows": int(count), "skipped": True}
    connection.execute(f"DROP TABLE IF EXISTS {table}")
    stay_map, _ = time_maps(stays)
    stay_ids = set(stay_map)
    item_ids = set(CHART_ITEM_NAMES)
    path = source_path(root, "icu", "chartevents.csv.gz")
    first_write = True
    rows = 0
    started = time.time()
    columns = ["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum", "valueuom"]
    for chunk_number, chunk in enumerate(
        pd.read_csv(path, usecols=columns, chunksize=chunksize, low_memory=False),
        start=1,
    ):
        chunk = chunk[
            chunk["stay_id"].isin(stay_ids)
            & chunk["itemid"].isin(item_ids)
            & chunk["valuenum"].notna()
        ].copy()
        if not chunk.empty:
            chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
            chunk["intime"] = chunk["stay_id"].map(stay_map)
            chunk["hour_from_icu"] = (
                (chunk["charttime"] - chunk["intime"]).dt.total_seconds() / 3600.0
            )
            chunk = chunk[chunk["hour_from_icu"].between(min_hour, max_hour)].copy()
            chunk["feature"] = chunk["itemid"].map(CHART_ITEM_NAMES)
            chunk = chunk[
                [
                    "subject_id",
                    "hadm_id",
                    "stay_id",
                    "charttime",
                    "hour_from_icu",
                    "itemid",
                    "feature",
                    "valuenum",
                    "valueuom",
                ]
            ]
            first_write = append_frame(connection, table, chunk, first_write)
            rows += len(chunk)
        if chunk_number % 25 == 0:
            print(
                f"chartevents chunks={chunk_number} selected_rows={rows} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    connection.execute(
        f"CREATE INDEX IF NOT EXISTS idx_chart_stay_time ON {table}(stay_id, hour_from_icu)"
    )
    connection.execute(
        f"CREATE INDEX IF NOT EXISTS idx_chart_feature ON {table}(feature)"
    )
    connection.commit()
    return {"rows": int(rows), "skipped": False}


def stream_labevents(
    connection: sqlite3.Connection,
    root: Path,
    stays: pd.DataFrame,
    chunksize: int,
    rebuild: bool,
    min_hour: float,
    max_hour: float,
) -> dict:
    table = "labevents_selected"
    if table_exists(connection, table) and not rebuild:
        count = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return {"rows": int(count), "skipped": True}
    connection.execute(f"DROP TABLE IF EXISTS {table}")
    _, hadm_map = time_maps(stays)
    hadm_ids = set(hadm_map)
    item_ids = set(LAB_ITEM_NAMES)
    path = source_path(root, "hosp", "labevents.csv.gz")
    first_write = True
    rows = 0
    started = time.time()
    columns = [
        "labevent_id",
        "subject_id",
        "hadm_id",
        "charttime",
        "itemid",
        "valuenum",
        "valueuom",
        "flag",
    ]
    for chunk_number, chunk in enumerate(
        pd.read_csv(path, usecols=columns, chunksize=chunksize, low_memory=False),
        start=1,
    ):
        chunk = chunk[
            chunk["hadm_id"].isin(hadm_ids)
            & chunk["itemid"].isin(item_ids)
            & chunk["valuenum"].notna()
        ].copy()
        if not chunk.empty:
            chunk["hadm_id"] = chunk["hadm_id"].astype(int)
            chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
            chunk["intime"] = chunk["hadm_id"].map(hadm_map)
            chunk["hour_from_icu"] = (
                (chunk["charttime"] - chunk["intime"]).dt.total_seconds() / 3600.0
            )
            chunk = chunk[chunk["hour_from_icu"].between(min_hour, max_hour)].copy()
            chunk["feature"] = chunk["itemid"].map(LAB_ITEM_NAMES)
            chunk = chunk[
                [
                    "labevent_id",
                    "subject_id",
                    "hadm_id",
                    "charttime",
                    "hour_from_icu",
                    "itemid",
                    "feature",
                    "valuenum",
                    "valueuom",
                    "flag",
                ]
            ]
            first_write = append_frame(connection, table, chunk, first_write)
            rows += len(chunk)
        if chunk_number % 25 == 0:
            print(
                f"labevents chunks={chunk_number} selected_rows={rows} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    connection.execute(
        f"CREATE INDEX IF NOT EXISTS idx_lab_hadm_time ON {table}(hadm_id, hour_from_icu)"
    )
    connection.execute(
        f"CREATE INDEX IF NOT EXISTS idx_lab_feature ON {table}(feature)"
    )
    connection.commit()
    return {"rows": int(rows), "skipped": False}


def stream_inputevents(
    connection: sqlite3.Connection,
    root: Path,
    stays: pd.DataFrame,
    chunksize: int,
    rebuild: bool,
    min_hour: float,
    max_hour: float,
) -> dict:
    table = "inputevents_selected"
    if table_exists(connection, table) and not rebuild:
        count = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return {"rows": int(count), "skipped": True}
    connection.execute(f"DROP TABLE IF EXISTS {table}")
    stay_map, _ = time_maps(stays)
    stay_ids = set(stay_map)
    item_to_group = {
        item_id: group
        for group, item_ids in INPUT_ITEM_GROUPS.items()
        for item_id in item_ids
    }
    path = source_path(root, "icu", "inputevents.csv.gz")
    first_write = True
    rows = 0
    started = time.time()
    columns = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "starttime",
        "endtime",
        "itemid",
        "amount",
        "amountuom",
        "rate",
        "rateuom",
        "statusdescription",
    ]
    for chunk_number, chunk in enumerate(
        pd.read_csv(path, usecols=columns, chunksize=chunksize, low_memory=False),
        start=1,
    ):
        chunk = chunk[
            chunk["stay_id"].isin(stay_ids)
            & chunk["itemid"].isin(item_to_group)
        ].copy()
        if not chunk.empty:
            valid_dose = (
                pd.to_numeric(chunk["amount"], errors="coerce").fillna(0).gt(0)
                | pd.to_numeric(chunk["rate"], errors="coerce").fillna(0).gt(0)
            )
            chunk = chunk[valid_dose].copy()
            chunk["starttime"] = pd.to_datetime(chunk["starttime"], errors="coerce")
            chunk["endtime"] = pd.to_datetime(chunk["endtime"], errors="coerce")
            chunk["intime"] = chunk["stay_id"].map(stay_map)
            chunk["start_hour"] = (
                (chunk["starttime"] - chunk["intime"]).dt.total_seconds() / 3600.0
            )
            chunk["end_hour"] = (
                (chunk["endtime"] - chunk["intime"]).dt.total_seconds() / 3600.0
            )
            chunk = chunk[chunk["start_hour"].between(min_hour, max_hour)].copy()
            chunk["treatment_group"] = chunk["itemid"].map(item_to_group)
            chunk = chunk[
                [
                    "subject_id",
                    "hadm_id",
                    "stay_id",
                    "starttime",
                    "endtime",
                    "start_hour",
                    "end_hour",
                    "itemid",
                    "treatment_group",
                    "amount",
                    "amountuom",
                    "rate",
                    "rateuom",
                    "statusdescription",
                ]
            ]
            first_write = append_frame(connection, table, chunk, first_write)
            rows += len(chunk)
        if chunk_number % 10 == 0:
            print(
                f"inputevents chunks={chunk_number} selected_rows={rows} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    connection.execute(
        f"CREATE INDEX IF NOT EXISTS idx_input_stay_time ON {table}(stay_id, start_hour)"
    )
    connection.execute(
        f"CREATE INDEX IF NOT EXISTS idx_input_group ON {table}(treatment_group)"
    )
    connection.commit()
    return {"rows": int(rows), "skipped": False}


def stream_outputevents(
    connection: sqlite3.Connection,
    root: Path,
    stays: pd.DataFrame,
    chunksize: int,
    rebuild: bool,
    min_hour: float,
    max_hour: float,
) -> dict:
    table = "outputevents_selected"
    if table_exists(connection, table) and not rebuild:
        count = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return {"rows": int(count), "skipped": True}
    connection.execute(f"DROP TABLE IF EXISTS {table}")
    stay_map, _ = time_maps(stays)
    stay_ids = set(stay_map)
    item_to_group = {
        item_id: group
        for group, item_ids in OUTPUT_ITEM_GROUPS.items()
        for item_id in item_ids
    }
    path = source_path(root, "icu", "outputevents.csv.gz")
    first_write = True
    rows = 0
    started = time.time()
    columns = ["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "value", "valueuom"]
    for chunk_number, chunk in enumerate(
        pd.read_csv(path, usecols=columns, chunksize=chunksize, low_memory=False),
        start=1,
    ):
        chunk = chunk[
            chunk["stay_id"].isin(stay_ids)
            & chunk["itemid"].isin(item_to_group)
            & pd.to_numeric(chunk["value"], errors="coerce").fillna(0).gt(0)
        ].copy()
        if not chunk.empty:
            chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
            chunk["intime"] = chunk["stay_id"].map(stay_map)
            chunk["hour_from_icu"] = (
                (chunk["charttime"] - chunk["intime"]).dt.total_seconds() / 3600.0
            )
            chunk = chunk[chunk["hour_from_icu"].between(min_hour, max_hour)].copy()
            chunk["output_group"] = chunk["itemid"].map(item_to_group)
            chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce")
            chunk = chunk[
                [
                    "subject_id",
                    "hadm_id",
                    "stay_id",
                    "charttime",
                    "hour_from_icu",
                    "itemid",
                    "output_group",
                    "value",
                    "valueuom",
                ]
            ]
            first_write = append_frame(connection, table, chunk, first_write)
            rows += len(chunk)
        if chunk_number % 10 == 0:
            print(
                f"outputevents chunks={chunk_number} selected_rows={rows} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    connection.execute(
        f"CREATE INDEX IF NOT EXISTS idx_output_stay_time ON {table}(stay_id, hour_from_icu)"
    )
    connection.commit()
    return {"rows": int(rows), "skipped": False}


def save_metadata(
    connection: sqlite3.Connection,
    database: Path,
    root: Path,
    stays: pd.DataFrame,
    table_results: dict,
    started: float,
    min_hour: float,
    max_hour: float,
) -> dict:
    metadata = {
        "database": str(database.resolve()),
        "mimic_root": str(root.resolve()),
        "mimic_version": "3.1",
        "cohort": "adult first ICU stay per patient",
        "n_stays": int(len(stays)),
        "event_window_hours": [float(min_hour), float(max_hour)],
        "table_results": table_results,
        "input_item_groups": {k: sorted(v) for k, v in INPUT_ITEM_GROUPS.items()},
        "chart_item_names": {str(k): v for k, v in CHART_ITEM_NAMES.items()},
        "lab_item_names": {str(k): v for k, v in LAB_ITEM_NAMES.items()},
        "output_item_groups": {k: sorted(v) for k, v in OUTPUT_ITEM_GROUPS.items()},
        "runtime_sec": float(time.time() - started),
    }
    connection.execute("DROP TABLE IF EXISTS build_metadata")
    pd.DataFrame(
        [{"key": key, "value": json.dumps(value)} for key, value in metadata.items()]
    ).to_sql("build_metadata", connection, if_exists="replace", index=False)
    connection.execute("ANALYZE")
    connection.commit()
    manifest_path = database.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", type=Path, default=DEFAULT_MIMIC_ROOT)
    parser.add_argument("--database", type=Path, default=PROJECT_ROOT / "data" / "mimic_iv_ccia.sqlite")
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--min-hour", type=float, default=-24.0)
    parser.add_argument("--max-hour", type=float, default=72.0)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--skip-chartevents",
        action="store_true",
        help="Skip the chartevents scan when the local source file is unavailable or incomplete.",
    )
    args = parser.parse_args()

    started = time.time()
    root = args.mimic_root.resolve()
    database = args.database.resolve()
    connection = connect_database(database)
    try:
        stays = write_base_tables(connection, root, args.rebuild)
        print(f"adult first ICU stays={len(stays)}", flush=True)
        results = {}
        results["inputevents_selected"] = stream_inputevents(
            connection,
            root,
            stays,
            args.chunksize,
            args.rebuild,
            args.min_hour,
            args.max_hour,
        )
        if args.skip_chartevents:
            results["chartevents_selected"] = {
                "rows": 0,
                "skipped": True,
                "reason": "explicit_skip",
            }
        else:
            results["chartevents_selected"] = stream_chartevents(
                connection,
                root,
                stays,
                args.chunksize,
                args.rebuild,
                args.min_hour,
                args.max_hour,
            )
        results["labevents_selected"] = stream_labevents(
            connection,
            root,
            stays,
            args.chunksize,
            args.rebuild,
            args.min_hour,
            args.max_hour,
        )
        results["outputevents_selected"] = stream_outputevents(
            connection,
            root,
            stays,
            args.chunksize,
            args.rebuild,
            args.min_hour,
            args.max_hour,
        )
        metadata = save_metadata(
            connection,
            database,
            root,
            stays,
            results,
            started,
            args.min_hour,
            args.max_hour,
        )
        print(json.dumps(metadata, indent=2), flush=True)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
