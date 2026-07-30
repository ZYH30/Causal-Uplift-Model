"""Target-trial cohort construction for early red-cell transfusion in MIMIC-IV."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from empirical_experiments.mimic_common import (
    clean_labs,
    diagnosis_flags,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MIMIC_ROOT = (
    PROJECT_ROOT.parent
    / "MIMIC-IV"
    / "physionet.org"
    / "files"
    / "mimiciv"
    / "3.1"
)

V3_EXCLUDED_FEATURES = {
    "days_since_last_admission_missing",
    "lab_bilirubin_total_last",
    "lab_bilirubin_total_last_missing",
    "lab_lymphocytes_last",
    "lab_lymphocytes_last_missing",
    "lab_magnesium_last",
    "lab_magnesium_last_missing",
    "lab_mcv_last",
    "lab_mcv_last_missing",
    "urine_measurements_6h",
    "urine_output_6h_missing",
}

SERVICE_GROUPS = {
    "service_medical": {"MED", "OMED"},
    "service_cardiac": {"CMED", "CSURG"},
    "service_surgical": {
        "SURG",
        "ORTHO",
        "VSURG",
        "TSURG",
        "PSURG",
        "GU",
        "GYN",
        "ENT",
        "DENT",
        "EYE",
    },
    "service_neuro": {"NMED", "NSURG"},
    "service_trauma": {"TRAUM"},
}


@dataclass(frozen=True)
class RBCTargetTrialSpec:
    trigger_low: float = 7.0
    trigger_high: float = 9.0
    trigger_start_hour: float = 0.0
    trigger_end_hour: float = 24.0
    grace_hours: float = 6.0
    baseline_lookback_hours: float = 24.0
    urine_lookback_hours: float = 6.0
    physiologic_start_hour: float = 6.0
    physiologic_end_hour: float = 24.0
    mortality_followup_days: float = 90.0

    def to_dict(self) -> dict:
        return asdict(self)


def _prior_diagnosis_history(
    stays: pd.DataFrame,
    mimic_root: Path,
) -> pd.DataFrame:
    admissions = pd.read_csv(
        mimic_root / "hosp" / "admissions.csv.gz",
        usecols=["subject_id", "hadm_id", "admittime", "dischtime"],
    )
    admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")
    admissions["dischtime"] = pd.to_datetime(admissions["dischtime"], errors="coerce")
    index = stays[["subject_id", "hadm_id", "admittime"]].rename(
        columns={
            "hadm_id": "index_hadm_id",
            "admittime": "index_admittime",
        }
    )
    previous = admissions.merge(index, on="subject_id", how="inner")
    previous = previous[
        previous["hadm_id"].ne(previous["index_hadm_id"])
        & previous["dischtime"].notna()
        & previous["dischtime"].le(previous["index_admittime"])
    ].copy()
    previous["days_since_discharge"] = (
        previous["index_admittime"] - previous["dischtime"]
    ).dt.total_seconds() / 86400.0
    utilization = previous.groupby("index_hadm_id", as_index=False).agg(
        prior_admission_count=("hadm_id", "nunique"),
        days_since_last_admission=("days_since_discharge", "min"),
    ).rename(columns={"index_hadm_id": "hadm_id"})

    prior_to_index = dict(
        zip(
            previous["hadm_id"].astype(int),
            previous["index_hadm_id"].astype(int),
        )
    )
    flag_chunks = []
    if prior_to_index:
        path = mimic_root / "hosp" / "diagnoses_icd.csv.gz"
        for chunk in pd.read_csv(
            path,
            usecols=["hadm_id", "icd_code", "icd_version"],
            chunksize=1_000_000,
        ):
            chunk = chunk[chunk["hadm_id"].isin(prior_to_index)].copy()
            if chunk.empty:
                continue
            chunk["hadm_id"] = chunk["hadm_id"].map(prior_to_index)
            flag_chunks.append(diagnosis_flags(chunk))
    if flag_chunks:
        flags = pd.concat(flag_chunks, ignore_index=True)
        flags = flags.groupby("hadm_id", as_index=False).max()
        flags = flags.rename(
            columns={
                column: column.replace("dx_", "hx_", 1)
                for column in flags.columns
                if column.startswith("dx_")
            }
        )
    else:
        flags = pd.DataFrame({"hadm_id": pd.Series(dtype=int)})
    return utilization.merge(flags, on="hadm_id", how="outer")


def read_rbc_source_tables(
    database: Path,
    mimic_root: Path = DEFAULT_MIMIC_ROOT,
) -> tuple[pd.DataFrame, ...]:
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
        outputs = pd.read_sql_query(
            """
            SELECT stay_id, hour_from_icu, output_group, value
            FROM outputevents_selected
            """,
            connection,
        )
    finally:
        connection.close()
    for column in ("admittime", "intime", "outtime"):
        stays[column] = pd.to_datetime(stays[column], errors="coerce")
    history = _prior_diagnosis_history(stays, mimic_root)
    services = pd.read_csv(
        mimic_root / "hosp" / "services.csv.gz",
        usecols=["hadm_id", "transfertime", "curr_service"],
    )
    services["transfertime"] = pd.to_datetime(
        services["transfertime"],
        errors="coerce",
    )
    procedures = pd.read_csv(
        mimic_root / "icu" / "procedureevents.csv.gz",
        usecols=[
            "hadm_id",
            "stay_id",
            "starttime",
            "itemid",
            "ordercategoryname",
        ],
    )
    procedures["starttime"] = pd.to_datetime(
        procedures["starttime"],
        errors="coerce",
    )
    item_dictionary = pd.read_csv(
        mimic_root / "icu" / "d_items.csv.gz",
        usecols=["itemid", "label", "category"],
    )
    procedures = procedures.merge(
        item_dictionary,
        on="itemid",
        how="left",
        validate="many_to_one",
    )
    return (
        stays,
        clean_labs(labs),
        inputs,
        outputs,
        history,
        services,
        procedures,
    )


def _first_anemia_trigger(
    labs: pd.DataFrame,
    spec: RBCTargetTrialSpec,
) -> pd.DataFrame:
    trigger = labs[
        labs["feature"].eq("hemoglobin")
        & labs["valuenum"].between(spec.trigger_low, spec.trigger_high)
        & labs["hour_from_icu"].between(
            spec.trigger_start_hour,
            spec.trigger_end_hour,
            inclusive="both",
        )
    ].copy()
    trigger = trigger.sort_values(["hadm_id", "hour_from_icu"])
    trigger = trigger.groupby("hadm_id", as_index=False).first()
    return trigger.rename(
        columns={
            "hour_from_icu": "trigger_hour",
            "valuenum": "trigger_value",
        }
    )[["hadm_id", "trigger_hour", "trigger_value"]]


def _rbc_treatment_summary(
    cohort: pd.DataFrame,
    inputs: pd.DataFrame,
    spec: RBCTargetTrialSpec,
) -> pd.DataFrame:
    events = inputs.loc[
        inputs["treatment_group"].eq("rbc"),
        ["stay_id", "start_hour", "amount"],
    ].copy()
    grid = cohort[["stay_id", "trigger_hour"]].merge(
        events,
        on="stay_id",
        how="left",
    )
    grid["relative_hour"] = grid["start_hour"] - grid["trigger_hour"]
    grid["prior_rbc"] = grid["relative_hour"].lt(0).fillna(False)
    grid["early_rbc"] = grid["relative_hour"].between(
        0,
        spec.grace_hours,
        inclusive="left",
    ).fillna(False)
    grid["late_rbc_24h"] = grid["relative_hour"].between(
        spec.grace_hours,
        spec.physiologic_end_hour,
        inclusive="both",
    ).fillna(False)
    grid["early_rbc_amount"] = np.where(
        grid["early_rbc"],
        pd.to_numeric(grid["amount"], errors="coerce").fillna(0),
        0,
    )
    return grid.groupby("stay_id", as_index=False).agg(
        prior_rbc=("prior_rbc", "max"),
        early_rbc=("early_rbc", "max"),
        late_rbc_24h=("late_rbc_24h", "max"),
        early_rbc_amount=("early_rbc_amount", "sum"),
    )


def _add_physiologic_outcome(
    cohort: pd.DataFrame,
    labs: pd.DataFrame,
    spec: RBCTargetTrialSpec,
) -> pd.DataFrame:
    followup = labs.loc[
        labs["feature"].eq("hemoglobin"),
        ["hadm_id", "hour_from_icu", "valuenum"],
    ].merge(
        cohort[["hadm_id", "trigger_hour"]],
        on="hadm_id",
        how="inner",
    )
    followup["relative_hour"] = (
        followup["hour_from_icu"] - followup["trigger_hour"]
    )
    followup = followup[
        followup["relative_hour"].between(
            spec.physiologic_start_hour,
            spec.physiologic_end_hour,
            inclusive="both",
        )
    ].copy()
    followup = followup.sort_values(["hadm_id", "relative_hour"])
    summary = followup.groupby("hadm_id", as_index=False).agg(
        followup_hb_first=("valuenum", "first"),
        followup_hb_mean=("valuenum", "mean"),
        followup_hb_min=("valuenum", "min"),
        followup_hb_max=("valuenum", "max"),
        followup_hb_count=("valuenum", "size"),
    )
    cohort = cohort.merge(summary, on="hadm_id", how="left")
    cohort["followup_hb_change"] = (
        cohort["followup_hb_mean"] - cohort["trigger_value"]
    )
    cohort["followup_hb_observed"] = (
        cohort["followup_hb_mean"].notna().astype(int)
    )
    return cohort


def build_rbc_landmark_cohort(
    stays: pd.DataFrame,
    labs: pd.DataFrame,
    inputs: pd.DataFrame,
    spec: RBCTargetTrialSpec | None = None,
) -> tuple[pd.DataFrame, dict]:
    spec = spec or RBCTargetTrialSpec()
    triggers = _first_anemia_trigger(labs, spec)
    cohort = triggers.merge(
        stays,
        on="hadm_id",
        how="inner",
        validate="one_to_one",
    )
    treatment = _rbc_treatment_summary(cohort, inputs, spec)
    cohort = cohort.merge(treatment, on="stay_id", how="left", validate="one_to_one")
    cohort[["prior_rbc", "early_rbc", "late_rbc_24h"]] = cohort[
        ["prior_rbc", "early_rbc", "late_rbc_24h"]
    ].fillna(False)
    cohort["early_rbc_amount"] = cohort["early_rbc_amount"].fillna(0.0)

    no_prior = cohort[~cohort["prior_rbc"]].copy()
    no_prior["landmark_hour"] = no_prior["trigger_hour"] + spec.grace_hours
    death_hour = pd.to_numeric(
        no_prior["days_to_death_from_icu"],
        errors="coerce",
    ) * 24.0
    remained_in_icu = pd.to_numeric(
        no_prior["icu_hours"],
        errors="coerce",
    ).ge(no_prior["landmark_hour"])
    alive_at_landmark = death_hour.isna() | death_hour.ge(no_prior["landmark_hour"])
    cohort = no_prior[remained_in_icu & alive_at_landmark].copy()

    death_hour = pd.to_numeric(
        cohort["days_to_death_from_icu"],
        errors="coerce",
    ) * 24.0
    followup_end = (
        cohort["landmark_hour"] + spec.mortality_followup_days * 24.0
    )
    cohort["Y"] = (
        death_hour.ge(cohort["landmark_hour"])
        & death_hour.le(followup_end)
    ).astype(int)
    cohort["T"] = cohort["early_rbc"].astype(int)
    cohort = _add_physiologic_outcome(cohort, labs, spec)

    flow = {
        "qualifying_first_anemia_triggers": int(len(triggers)),
        "linked_adult_first_icu_stays": int(len(treatment)),
        "excluded_prior_rbc": int(treatment["prior_rbc"].sum()),
        "eligible_before_landmark_requirement": int(len(no_prior)),
        "excluded_not_in_icu_or_dead_at_landmark": int(len(no_prior) - len(cohort)),
        "landmark_eligible": int(len(cohort)),
        "treated_within_grace_window": int(cohort["T"].sum()),
        "control_at_landmark": int((1 - cohort["T"]).sum()),
        "late_rbc_among_controls": int(
            ((cohort["T"] == 0) & cohort["late_rbc_24h"]).sum()
        ),
        "mortality_90d_events": int(cohort["Y"].sum()),
        "followup_hb_observed": int(cohort["followup_hb_observed"].sum()),
    }
    return cohort.reset_index(drop=True), flow


def _demographic_features(cohort: pd.DataFrame) -> pd.DataFrame:
    frame = cohort[
        [
            "hadm_id",
            "age",
            "gender",
            "race",
            "insurance",
            "first_careunit",
            "admission_type",
            "admission_location",
            "anchor_year_group",
        ]
    ].copy()
    frame["sex_male"] = frame["gender"].astype(str).str.upper().eq("M").astype(int)
    race = frame["race"].fillna("UNKNOWN").astype(str).str.upper()
    frame["race_white"] = race.str.contains("WHITE").astype(int)
    frame["race_black"] = race.str.contains("BLACK").astype(int)
    frame["race_hispanic"] = race.str.contains(
        "HISPANIC|LATINO",
        regex=True,
    ).astype(int)
    frame["race_asian"] = race.str.contains("ASIAN").astype(int)
    insurance = frame["insurance"].fillna("UNKNOWN").astype(str).str.upper()
    frame["insurance_medicare"] = insurance.str.contains("MEDICARE").astype(int)
    frame["insurance_medicaid"] = insurance.str.contains("MEDICAID").astype(int)
    frame["insurance_private"] = insurance.str.contains("PRIVATE").astype(int)
    unit = frame["first_careunit"].fillna("UNKNOWN").astype(str).str.upper()
    frame["unit_cardiac"] = unit.str.contains(
        "CORONARY|CARDIAC|CVICU",
        regex=True,
    ).astype(int)
    frame["unit_medical"] = unit.str.contains("MEDICAL").astype(int)
    frame["unit_surgical"] = unit.str.contains(
        "SURGICAL|TRAUMA",
        regex=True,
    ).astype(int)
    frame["unit_neuro"] = unit.str.contains("NEURO").astype(int)
    admission = frame["admission_type"].fillna("UNKNOWN").astype(str).str.upper()
    frame["admission_emergency"] = admission.str.contains(
        "EMERGENCY|URGENT",
        regex=True,
    ).astype(int)
    location = frame["admission_location"].fillna("UNKNOWN").astype(str).str.upper()
    frame["admission_from_ed"] = location.str.contains(
        "EMERGENCY",
        regex=True,
    ).astype(int)
    frame["admission_transfer"] = location.str.contains(
        "TRANSFER|CLINIC",
        regex=True,
    ).astype(int)
    era_order = {
        "2008 - 2010": 0,
        "2011 - 2013": 1,
        "2014 - 2016": 2,
        "2017 - 2019": 3,
        "2020 - 2022": 4,
    }
    frame["era_index"] = frame["anchor_year_group"].map(era_order).astype(float)
    keep = [
        "hadm_id",
        "age",
        "sex_male",
        "race_white",
        "race_black",
        "race_hispanic",
        "race_asian",
        "insurance_medicare",
        "insurance_medicaid",
        "insurance_private",
        "unit_cardiac",
        "unit_medical",
        "unit_surgical",
        "unit_neuro",
        "admission_emergency",
        "admission_from_ed",
        "admission_transfer",
        "era_index",
    ]
    return frame[keep]


def _baseline_lab_features(
    cohort: pd.DataFrame,
    labs: pd.DataFrame,
    spec: RBCTargetTrialSpec,
) -> pd.DataFrame:
    baseline = labs.merge(
        cohort[["hadm_id", "trigger_hour"]],
        on="hadm_id",
        how="inner",
    )
    baseline = baseline[
        baseline["hour_from_icu"].between(
            baseline["trigger_hour"] - spec.baseline_lookback_hours,
            baseline["trigger_hour"],
            inclusive="both",
        )
    ].copy()
    baseline = baseline.sort_values(["hadm_id", "feature", "hour_from_icu"])
    last = baseline.groupby(["hadm_id", "feature"], as_index=False).tail(1)
    pivot = last.pivot(index="hadm_id", columns="feature", values="valuenum")
    pivot.columns = [f"lab_{column}_last" for column in pivot.columns]
    pivot = pivot.reset_index()

    intensity = baseline.groupby("hadm_id", as_index=False).agg(
        baseline_lab_count=("valuenum", "size"),
    )
    hb = baseline[baseline["feature"].eq("hemoglobin")].groupby(
        "hadm_id",
        as_index=False,
    ).agg(
        baseline_hb_count=("valuenum", "size"),
        baseline_hb_min=("valuenum", "min"),
    )
    return pivot.merge(intensity, on="hadm_id", how="outer").merge(
        hb,
        on="hadm_id",
        how="left",
    )


def _baseline_treatment_features(
    cohort: pd.DataFrame,
    inputs: pd.DataFrame,
    spec: RBCTargetTrialSpec,
) -> pd.DataFrame:
    baseline = inputs.merge(
        cohort[["stay_id", "hadm_id", "trigger_hour"]],
        on=["stay_id", "hadm_id"],
        how="inner",
    )
    baseline = baseline[
        baseline["start_hour"].between(
            baseline["trigger_hour"] - spec.baseline_lookback_hours,
            baseline["trigger_hour"],
            inclusive="left",
        )
    ].copy()
    baseline = baseline[~baseline["treatment_group"].eq("rbc")]
    baseline["present"] = 1
    summary = baseline.groupby(
        ["hadm_id", "treatment_group"],
        as_index=False,
    ).agg(present=("present", "max"))
    pivot = summary.pivot(
        index="hadm_id",
        columns="treatment_group",
        values="present",
    ).fillna(0)
    pivot.columns = [f"prior_{column}" for column in pivot.columns]
    return pivot.reset_index()


def _baseline_urine_features(
    cohort: pd.DataFrame,
    outputs: pd.DataFrame,
    spec: RBCTargetTrialSpec,
) -> pd.DataFrame:
    urine = outputs[outputs["output_group"].eq("urine")].merge(
        cohort[["stay_id", "hadm_id", "trigger_hour"]],
        on="stay_id",
        how="inner",
    )
    urine = urine[
        urine["hour_from_icu"].between(
            urine["trigger_hour"] - spec.urine_lookback_hours,
            urine["trigger_hour"],
            inclusive="both",
        )
    ].copy()
    return urine.groupby("hadm_id", as_index=False).agg(
        urine_output_6h=("value", "sum"),
        urine_measurements_6h=("value", "size"),
    )


def _baseline_service_features(
    cohort: pd.DataFrame,
    services: pd.DataFrame,
) -> pd.DataFrame:
    index = cohort[["hadm_id", "intime", "trigger_hour"]].copy()
    index["trigger_time"] = index["intime"] + pd.to_timedelta(
        index["trigger_hour"],
        unit="h",
    )
    prior = services.merge(
        index[["hadm_id", "trigger_time"]],
        on="hadm_id",
        how="inner",
    )
    prior = prior[
        prior["transfertime"].notna()
        & prior["transfertime"].le(prior["trigger_time"])
    ].sort_values(["hadm_id", "transfertime"])
    latest = prior.groupby("hadm_id", as_index=False).tail(1)
    latest = index[["hadm_id"]].merge(
        latest[["hadm_id", "curr_service"]],
        on="hadm_id",
        how="left",
    )
    service = latest["curr_service"].fillna("UNKNOWN").astype(str).str.upper()
    for feature, codes in SERVICE_GROUPS.items():
        latest[feature] = service.isin(codes).astype(int)
    return latest[["hadm_id", *SERVICE_GROUPS]]


def _baseline_procedure_features(
    cohort: pd.DataFrame,
    procedures: pd.DataFrame,
    spec: RBCTargetTrialSpec,
) -> pd.DataFrame:
    index = cohort[["hadm_id", "stay_id", "intime", "trigger_hour"]].copy()
    index["trigger_time"] = index["intime"] + pd.to_timedelta(
        index["trigger_hour"],
        unit="h",
    )
    index["baseline_start"] = index["trigger_time"] - pd.to_timedelta(
        spec.baseline_lookback_hours,
        unit="h",
    )
    prior = procedures.merge(
        index[["hadm_id", "stay_id", "trigger_time", "baseline_start"]],
        on=["hadm_id", "stay_id"],
        how="inner",
    )
    prior = prior[
        prior["starttime"].notna()
        & prior["starttime"].ge(prior["baseline_start"])
        & prior["starttime"].lt(prior["trigger_time"])
    ].copy()
    label = prior["label"].fillna("").astype(str)
    category = prior["category"].fillna("").astype(str)
    order_category = prior["ordercategoryname"].fillna("").astype(str)
    prior["pretrigger_invasive_respiratory_support"] = (
        label.str.contains(
            "Invasive Ventilation|Intubation|Tracheostomy",
            case=False,
            regex=True,
        )
        & ~label.str.contains("Extubation|Non-invasive", case=False, regex=True)
    ).astype(int)
    prior["pretrigger_noninvasive_ventilation"] = label.str.contains(
        "Non-invasive Ventilation",
        case=False,
        regex=False,
    ).astype(int)
    prior["pretrigger_dialysis"] = (
        category.str.contains("Dialysis", case=False, regex=False)
        | order_category.str.contains("Dialysis", case=False, regex=False)
        | label.str.contains(
            "Hemodialysis|Peritoneal Dialysis",
            case=False,
            regex=True,
        )
    ).astype(int)
    prior["pretrigger_invasive_line"] = category.eq(
        "Access Lines - Invasive"
    ).astype(int)
    prior["pretrigger_operating_room"] = label.isin(
        {"OR Sent", "OR Received"}
    ).astype(int)
    prior["pretrigger_mechanical_support"] = label.str.contains(
        "ECMO|IABP|Impella|Centrimag|Tandem Heart",
        case=False,
        regex=True,
    ).astype(int)
    binary_features = [
        "pretrigger_invasive_respiratory_support",
        "pretrigger_noninvasive_ventilation",
        "pretrigger_dialysis",
        "pretrigger_invasive_line",
        "pretrigger_operating_room",
        "pretrigger_mechanical_support",
    ]
    if prior.empty:
        summary = pd.DataFrame(columns=["hadm_id", *binary_features])
        counts = pd.DataFrame(columns=["hadm_id", "pretrigger_procedure_count"])
    else:
        summary = prior.groupby("hadm_id", as_index=False)[binary_features].max()
        counts = prior.groupby("hadm_id", as_index=False).agg(
            pretrigger_procedure_count=("itemid", "size"),
        )
    result = index[["hadm_id"]].merge(summary, on="hadm_id", how="left")
    result = result.merge(counts, on="hadm_id", how="left")
    result[binary_features + ["pretrigger_procedure_count"]] = result[
        binary_features + ["pretrigger_procedure_count"]
    ].fillna(0)
    return result


def build_rbc_baseline_frame(
    cohort: pd.DataFrame,
    labs: pd.DataFrame,
    inputs: pd.DataFrame,
    outputs: pd.DataFrame,
    history: pd.DataFrame,
    services: pd.DataFrame,
    procedures: pd.DataFrame,
    spec: RBCTargetTrialSpec | None = None,
) -> pd.DataFrame:
    spec = spec or RBCTargetTrialSpec()
    frame = cohort[
        [
            "subject_id",
            "hadm_id",
            "stay_id",
            "anchor_year_group",
            "trigger_hour",
            "trigger_value",
            "landmark_hour",
            "T",
            "Y",
            "late_rbc_24h",
            "early_rbc_amount",
            "followup_hb_change",
            "followup_hb_observed",
        ]
    ].copy()
    frame = frame.merge(_demographic_features(cohort), on="hadm_id", how="left")
    frame = frame.merge(history, on="hadm_id", how="left")
    frame = frame.merge(
        _baseline_lab_features(cohort, labs, spec),
        on="hadm_id",
        how="left",
    )
    frame = frame.merge(
        _baseline_treatment_features(cohort, inputs, spec),
        on="hadm_id",
        how="left",
    )
    frame = frame.merge(
        _baseline_urine_features(cohort, outputs, spec),
        on="hadm_id",
        how="left",
    )
    frame = frame.merge(
        _baseline_service_features(cohort, services),
        on="hadm_id",
        how="left",
    )
    frame = frame.merge(
        _baseline_procedure_features(cohort, procedures, spec),
        on="hadm_id",
        how="left",
    )

    history_cols = [column for column in frame if column.startswith("hx_")]
    prior_cols = [column for column in frame if column.startswith("prior_")]
    frame[history_cols + prior_cols] = frame[history_cols + prior_cols].fillna(0)
    frame["days_since_last_admission_missing"] = (
        frame["days_since_last_admission"].isna().astype(int)
    )
    frame["urine_observed_6h"] = frame["urine_output_6h"].notna().astype(int)

    lab_cols = [column for column in frame if column.startswith("lab_")]
    for column in lab_cols + ["urine_output_6h"]:
        frame[f"{column}_missing"] = frame[column].isna().astype(int)

    # The trigger is the last qualifying hemoglobin measurement, so retaining
    # both variables would duplicate the same clinical signal.
    frame = frame.drop(columns=["lab_hemoglobin_last"], errors="ignore")
    return frame


def candidate_feature_manifest(
    frame: pd.DataFrame,
    discovery_mask: pd.Series,
    max_missing: float = 0.70,
) -> tuple[list[str], pd.DataFrame]:
    excluded = {
        "subject_id",
        "hadm_id",
        "stay_id",
        "anchor_year_group",
        "landmark_hour",
        "T",
        "Y",
        "late_rbc_24h",
        "early_rbc_amount",
        "followup_hb_change",
        "followup_hb_observed",
    }
    discovery = frame.loc[discovery_mask]
    rows = []
    retained = []
    for column in frame.columns:
        if column in excluded:
            continue
        values = pd.to_numeric(discovery[column], errors="coerce")
        missing_rate = float(values.isna().mean())
        unique_observed = int(values.nunique(dropna=True))
        keep = (
            missing_rate <= max_missing
            and unique_observed > 1
            and column not in V3_EXCLUDED_FEATURES
        )
        rows.append(
            {
                "feature": column,
                "discovery_missing_rate": missing_rate,
                "discovery_unique_observed": unique_observed,
                "retained": keep,
                "exclusion_reason": (
                    "prespecified_redundant_or_low_value"
                    if column in V3_EXCLUDED_FEATURES
                    else (
                        "discovery_missingness"
                        if missing_rate > max_missing
                        else (
                            "no_discovery_variation"
                            if unique_observed <= 1
                            else ""
                        )
                    )
                ),
            }
        )
        if keep:
            retained.append(column)
    return retained, pd.DataFrame(rows)


def impute_from_training(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    train_out = train.copy()
    eval_out = evaluation.copy()
    medians = {}
    for column in feature_cols:
        train_values = pd.to_numeric(train_out[column], errors="coerce")
        eval_values = pd.to_numeric(eval_out[column], errors="coerce")
        median = float(train_values.median()) if train_values.notna().any() else 0.0
        medians[column] = median
        train_out[column] = train_values.fillna(median)
        eval_out[column] = eval_values.fillna(median)
    return train_out, eval_out, medians
