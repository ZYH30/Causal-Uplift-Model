"""Shared MIMIC-IV laboratory cleaning and diagnosis feature definitions."""

from __future__ import annotations

import numpy as np
import pandas as pd


CANONICAL_LABS = {
    "hemoglobin_blood_gas": "hemoglobin",
    "hemoglobin_chemistry": "hemoglobin",
    "lactate_blood_gas": "lactate",
    "lactate_blood_gas_new": "lactate",
    "creatinine_new": "creatinine",
    "glucose_new": "glucose",
    "potassium_new": "potassium",
    "sodium_new": "sodium",
    "bun_new": "bun",
    "ptt_new": "ptt",
    "albumin_new": "albumin",
    "platelets_new": "platelets",
}

PLAUSIBLE_RANGES = {
    "albumin": (0.5, 7.0),
    "bicarbonate": (2.0, 60.0),
    "bilirubin_total": (0.0, 100.0),
    "bun": (1.0, 300.0),
    "creatinine": (0.1, 40.0),
    "glucose": (20.0, 1000.0),
    "hematocrit": (5.0, 75.0),
    "hemoglobin": (2.0, 25.0),
    "inr": (0.5, 30.0),
    "lactate": (0.1, 40.0),
    "lymphocytes": (0.0, 100.0),
    "magnesium": (0.3, 15.0),
    "mcv": (40.0, 160.0),
    "platelets": (1.0, 3000.0),
    "potassium": (1.0, 10.0),
    "pt": (5.0, 160.0),
    "ptt": (5.0, 180.0),
    "sodium": (90.0, 190.0),
    "wbc": (0.1, 300.0),
}


def clean_labs(labs: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize selected laboratory names and remove implausible values."""

    labs = labs.copy()
    labs["feature"] = labs["feature"].replace(CANONICAL_LABS)
    labs["valuenum"] = pd.to_numeric(labs["valuenum"], errors="coerce")
    valid = np.ones(len(labs), dtype=bool)
    for feature, (low, high) in PLAUSIBLE_RANGES.items():
        mask = labs["feature"].eq(feature)
        valid[mask.to_numpy()] = (
            labs.loc[mask, "valuenum"].between(low, high).to_numpy()
        )
    return labs[valid & labs["valuenum"].notna()].copy()


def diagnosis_flags(diagnoses: pd.DataFrame) -> pd.DataFrame:
    """Map ICD-9/10 codes to the prespecified baseline condition groups."""

    diagnoses = diagnoses.copy()
    diagnoses["code"] = (
        diagnoses["icd_code"]
        .astype(str)
        .str.upper()
        .str.replace(".", "", regex=False)
    )
    version10 = diagnoses["icd_version"].eq(10)
    version9 = diagnoses["icd_version"].eq(9)
    numeric_prefix = pd.to_numeric(
        diagnoses["code"].str[:3],
        errors="coerce",
    )
    definitions = {
        "dx_sepsis": (
            version10 & diagnoses["code"].str.startswith(("A40", "A41", "R652"))
        )
        | (
            version9
            & diagnoses["code"].str.startswith(
                ("038", "99591", "99592", "78552")
            )
        ),
        "dx_heart_failure": (
            version10 & diagnoses["code"].str.startswith("I50")
        )
        | (version9 & diagnoses["code"].str.startswith("428")),
        "dx_renal": (
            version10 & diagnoses["code"].str.startswith(("N17", "N18", "N19"))
        )
        | (
            version9 & diagnoses["code"].str.startswith(("584", "585", "586"))
        ),
        "dx_liver": (
            version10
            & diagnoses["code"].str.startswith(("K70", "K71", "K72", "K74", "K76"))
        )
        | (
            version9
            & diagnoses["code"].str.startswith(("570", "571", "572", "573"))
        ),
        "dx_malignancy": (
            version10 & diagnoses["code"].str.startswith("C")
        )
        | (version9 & numeric_prefix.between(140, 209)),
        "dx_diabetes": (
            version10
            & diagnoses["code"].str.startswith(("E08", "E09", "E10", "E11", "E13"))
        )
        | (version9 & diagnoses["code"].str.startswith("250")),
        "dx_anemia": (
            version10
            & diagnoses["code"].str.startswith(
                (
                    "D50",
                    "D51",
                    "D52",
                    "D53",
                    "D55",
                    "D56",
                    "D57",
                    "D58",
                    "D59",
                    "D60",
                    "D61",
                    "D62",
                    "D63",
                    "D64",
                )
            )
        )
        | (version9 & numeric_prefix.between(280, 285)),
        "dx_bleeding": (
            version10 & diagnoses["code"].str.startswith(("K92", "D62", "R58"))
        )
        | (
            version9
            & diagnoses["code"].str.startswith(("578", "2851", "4590"))
        ),
        "dx_ischemic_heart": (
            version10
            & diagnoses["code"].str.startswith(
                ("I20", "I21", "I22", "I23", "I24", "I25")
            )
        )
        | (
            version9
            & diagnoses["code"].str.startswith(
                ("410", "411", "412", "413", "414")
            )
        ),
        "dx_chronic_lung": (
            version10
            & diagnoses["code"].str.startswith(
                ("J40", "J41", "J42", "J43", "J44", "J45", "J46", "J47")
            )
        )
        | (version9 & numeric_prefix.between(490, 496)),
    }
    flag_frame = pd.DataFrame({"hadm_id": diagnoses["hadm_id"]})
    for name, mask in definitions.items():
        flag_frame[name] = mask.astype(np.int8)
    return flag_frame.groupby("hadm_id", as_index=False).max()
