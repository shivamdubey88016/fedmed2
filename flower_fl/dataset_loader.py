"""
Healthcare Dataset Loader for Flower Federated Learning
Loads patient records from MIMIC-IV cohort CSV files or pandas DataFrames.
Supports dynamic addition of custom hospital cohorts (via CSV upload or custom synthesis).
"""

import os
import glob
import csv
import random
from typing import Dict, Tuple, List, Any, Optional, Union
import pandas as pd
import numpy as np
import torch

from fl_engine.dataset import BENCHMARK_DATASETS_META, generate_benchmark_dataset_records

FEATURE_NAMES = [
    "norm_gender",
    "norm_age",
    "norm_ethnicity",
    "norm_hr",
    "norm_sbp",
    "norm_glucose",
    "norm_creatinine",
    "norm_o2",
    "norm_lactate",
    "norm_comorb"
]

def load_preset_benchmark_cohort(
    dataset_key: str = "mimic_iv",
    count: int = 250,
    seed: int = 42
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Loads one of the top 5 clinical benchmark datasets from the original website/system:
    - mimic_iv: MIMIC-IV ICU EHR (~380k records)
    - eicu: eICU Multi-Center Critical Care (~200k stays)
    - medmnist: MedMNIST Medical Imaging representation proxies
    - covid19_imaging: COVID-19 Pulmonary/Hypoxia Cohort (~50k+ images)
    - tcga: The Cancer Genome Atlas Oncological Cohort (~11k patients)
    
    Returns: (parsed_tensor_dict, pandas_dataframe)
    """
    if dataset_key not in BENCHMARK_DATASETS_META:
        dataset_key = "mimic_iv"

    records = generate_benchmark_dataset_records(dataset_key=dataset_key, count=count, seed=seed)
    df_rows = []
    for r in records:
        raw = r["raw_data"]
        df_rows.append({
            "gender": "Female" if r["gender"] == 0 else "Male",
            "age": raw["age"],
            "ethnicity": f"Group_{chr(65 + r['ethnicity'])}",
            "heart_rate": raw["heart_rate"],
            "systolic_bp": raw["systolic_bp"],
            "glucose": raw["glucose"],
            "creatinine": raw["creatinine"],
            "oxygen_sat": raw["oxygen_sat"],
            "lactate": raw["lactate"],
            "comorbidity_score": raw["comorbidity_score"],
            "label": r["label"]
        })
    df = pd.DataFrame(df_rows)
    parsed = parse_patient_dataframe(df)
    return parsed, df

def parse_patient_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Parses a pandas DataFrame into standardized PyTorch tensors and demographic metadata.
    Handles flexible column naming conventions (case-insensitive).
    """
    # Lowercase all column names
    col_map = {c: str(c).strip().lower() for c in df.columns}
    clean_df = df.rename(columns=col_map)

    features_list: List[List[float]] = []
    labels_list: List[int] = []
    gender_list: List[int] = []
    age_group_list: List[int] = []
    ethnicity_list: List[int] = []
    raw_records: List[Dict[str, Any]] = []

    for _, row in clean_df.iterrows():
        # Gender parsing (0: Female, 1: Male)
        g_raw = str(row.get("gender", "0")).strip().lower()
        gender = 0 if g_raw in ["0", "0.0", "f", "female", "w", "woman"] else 1

        # Age & Age Group
        try:
            age = float(row.get("age", 55.0))
        except (ValueError, TypeError):
            age = 55.0
        age_group = 1 if age >= 60.0 else 0

        # Ethnicity parsing (0: Group A, 1: Group B, 2: Group C)
        eth_raw = str(row.get("ethnicity", "0")).strip().lower()
        if eth_raw in ["0", "a", "group_a", "alpha", "group a"]:
            ethnicity = 0
        elif eth_raw in ["1", "b", "group_b", "beta", "group b"]:
            ethnicity = 1
        elif eth_raw in ["2", "c", "group_c", "gamma", "group c"]:
            ethnicity = 2
        else:
            try:
                ethnicity = int(float(eth_raw)) % 3
            except (ValueError, TypeError):
                ethnicity = 0

        # Clinical Vitals with safe defaults
        try: heart_rate = float(row.get("heart_rate", 80.0))
        except: heart_rate = 80.0
        try: systolic_bp = float(row.get("systolic_bp", 120.0))
        except: systolic_bp = 120.0
        try: glucose = float(row.get("glucose", 110.0))
        except: glucose = 110.0
        try: creatinine = float(row.get("creatinine", 1.0))
        except: creatinine = 1.0
        try: oxygen_sat = float(row.get("oxygen_sat", 96.0))
        except: oxygen_sat = 96.0
        try: lactate = float(row.get("lactate", 1.5))
        except: lactate = 1.5
        try: comorbidity_score = float(row.get("comorbidity_score", 1.0))
        except: comorbidity_score = 1.0

        # Target label
        lbl_raw = str(row.get("label", row.get("target", row.get("outcome", "0")))).strip().lower()
        label = 1 if lbl_raw in ["1", "1.0", "true", "yes", "positive", "high", "death", "mortality"] else 0

        # Standardized z-score normalization
        norm_gender = 1.0 if gender == 1 else -1.0
        norm_age = (age - 55.0) / 18.0
        norm_ethnicity = float(ethnicity - 1.0)
        norm_hr = (heart_rate - 80.0) / 15.0
        norm_sbp = (systolic_bp - 125.0) / 20.0
        norm_glucose = (glucose - 110.0) / 35.0
        norm_creatinine = (creatinine - 1.0) / 0.5
        norm_o2 = (95.0 - oxygen_sat) / 4.0
        norm_lactate = (lactate - 1.5) / 0.8
        norm_comorb = (comorbidity_score - 2.5) / 1.5

        feat = [
            norm_gender, norm_age, norm_ethnicity, norm_hr, norm_sbp,
            norm_glucose, norm_creatinine, norm_o2, norm_lactate, norm_comorb
        ]

        features_list.append(feat)
        labels_list.append(label)
        gender_list.append(gender)
        age_group_list.append(age_group)
        ethnicity_list.append(ethnicity)
        raw_records.append({
            "gender": "Female" if gender == 0 else "Male",
            "age": int(round(age)),
            "ethnicity": f"Group_{chr(65 + ethnicity)}",
            "heart_rate": round(heart_rate, 1),
            "systolic_bp": round(systolic_bp, 1),
            "glucose": round(glucose, 1),
            "creatinine": round(creatinine, 2),
            "oxygen_sat": round(oxygen_sat, 1),
            "lactate": round(lactate, 2),
            "comorbidity_score": int(round(comorbidity_score)),
            "label": label
        })

    X = torch.tensor(features_list, dtype=torch.float32)
    y = torch.tensor(labels_list, dtype=torch.float32)
    g = torch.tensor(gender_list, dtype=torch.long)
    a = torch.tensor(age_group_list, dtype=torch.long)
    e = torch.tensor(ethnicity_list, dtype=torch.long)

    return {
        "X": X,
        "y": y,
        "gender": g,
        "age_group": a,
        "ethnicity": e,
        "raw_records": raw_records,
        "sample_size": len(labels_list)
    }

def parse_patient_csv(csv_path: str) -> Dict[str, Any]:
    """Parses patient records from a CSV file path."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    return parse_patient_dataframe(df)

def load_all_hospital_datasets(
    data_dir: str = "data",
    custom_hospitals: Optional[Dict[str, Dict[str, Any]]] = None
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Loads all discovered hospital datasets from data_dir plus any custom hospitals in memory.
    Discovers hospital_a.csv, hospital_b.csv, hospital_c.csv, and any hospital_*.csv files.
    """
    hospitals: Dict[str, Dict[str, Any]] = {}

    # 1. Discover all hospital CSVs in data_dir
    if os.path.exists(data_dir):
        hospital_files = sorted(glob.glob(os.path.join(data_dir, "hospital_*.csv")))
        for fpath in hospital_files:
            fname = os.path.basename(fpath)
            hid = fname.replace(".csv", "").lower()
            try:
                hospitals[hid] = parse_patient_csv(fpath)
            except Exception as e:
                print(f"Warning: Could not parse {fpath}: {e}")

    # Fallback default if directory empty
    if not hospitals:
        for hid in ["hospital_a", "hospital_b", "hospital_c"]:
            fpath = os.path.join(data_dir, f"{hid}.csv")
            if os.path.exists(fpath):
                hospitals[hid] = parse_patient_csv(fpath)

    # 2. Merge user-provided in-memory custom hospitals
    if custom_hospitals:
        for hid, h_data in custom_hospitals.items():
            hospitals[hid] = h_data

    # 3. Load holdout global evaluation cohort
    test_path = os.path.join(data_dir, "global_evaluation_cohort.csv")
    if os.path.exists(test_path):
        global_test = parse_patient_csv(test_path)
    else:
        # Fallback: synthesize a default evaluation cohort
        first_h = next(iter(hospitals.values()))
        global_test = first_h

    return hospitals, global_test

def generate_synthetic_custom_hospital(
    hospital_name: str = "Hospital D (Custom Center)",
    sample_size: int = 1000,
    female_pct: float = 0.50,
    geriatric_pct: float = 0.50,
    bias_shift: float = 0.0,
    seed: int = 123
) -> pd.DataFrame:
    """
    Generates a realistic synthetic hospital EHR dataset with user-specified demographic shifts.
    """
    rng = random.Random(seed)
    records = []

    for _ in range(sample_size):
        # Demographic distribution
        gender_num = 0 if rng.random() < female_pct else 1
        age_group = 1 if rng.random() < geriatric_pct else 0
        eth_roll = rng.random()
        ethnicity = 0 if eth_roll < 0.50 else (1 if eth_roll < 0.80 else 2)

        age = rng.randint(20, 58) if age_group == 0 else rng.randint(60, 88)
        heart_rate = round(rng.gauss(82.0, 14.0), 1)
        systolic_bp = round(rng.gauss(128.0, 18.0), 1)
        glucose = round(rng.gauss(115.0, 32.0), 1)
        creatinine = round(max(0.4, rng.gauss(1.1, 0.4)), 2)
        oxygen_sat = round(min(100.0, max(80.0, rng.gauss(96.0, 3.2))), 1)
        lactate = round(max(0.5, rng.gauss(1.8, 0.8)), 2)
        comorbidity_score = rng.randint(0, 4) if age_group == 0 else rng.randint(1, 6)

        # Normalized variables for ground-truth logit
        norm_gender = 1.0 if gender_num == 1 else -1.0
        norm_age = (age - 55.0) / 18.0
        norm_hr = (heart_rate - 80.0) / 15.0
        norm_sbp = (systolic_bp - 125.0) / 20.0
        norm_glucose = (glucose - 110.0) / 35.0
        norm_creatinine = (creatinine - 1.0) / 0.5
        norm_o2 = (95.0 - oxygen_sat) / 4.0
        norm_lactate = (lactate - 1.5) / 0.8
        norm_comorb = (comorbidity_score - 2.5) / 1.5

        # Ground truth mortality risk
        logit = (
            -0.80
            + 0.55 * norm_age
            + 0.35 * norm_hr
            + 0.40 * norm_sbp
            + 0.30 * norm_glucose
            + 0.50 * norm_creatinine
            + 0.60 * norm_o2
            + 0.65 * norm_lactate
            + 0.45 * norm_comorb
            + bias_shift
        )
        prob = 1.0 / (1.0 + np.exp(-logit))
        label = 1 if rng.random() < prob else 0

        records.append({
            "gender": "Female" if gender_num == 0 else "Male",
            "age": age,
            "ethnicity": f"Group_{chr(65 + ethnicity)}",
            "heart_rate": heart_rate,
            "systolic_bp": systolic_bp,
            "glucose": glucose,
            "creatinine": creatinine,
            "oxygen_sat": oxygen_sat,
            "lactate": lactate,
            "comorbidity_score": comorbidity_score,
            "label": label
        })

    return pd.DataFrame(records)

def save_custom_hospital_to_disk(
    df: pd.DataFrame,
    hospital_id: str,
    data_dir: str = "data"
) -> str:
    """Saves custom hospital DataFrame to data/{hospital_id}.csv."""
    os.makedirs(data_dir, exist_ok=True)
    clean_id = hospital_id.lower().replace(" ", "_")
    if not clean_id.startswith("hospital_"):
        clean_id = f"hospital_{clean_id}"
    file_path = os.path.join(data_dir, f"{clean_id}.csv")
    df.to_csv(file_path, index=False, encoding="utf-8")
    return file_path
