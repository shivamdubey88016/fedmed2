"""
Benchmark Healthcare Dataset Generator for Federated Learning
Simulates non-IID Electronic Health Record (EHR) cohorts across 3 hospital institutions
incorporating demographic attributes (Gender, Age group, Ethnicity) and clinical indicators
(Heart rate, Systolic BP, Glucose, Creatinine, Oxygen Saturation, Lactate, Comorbidity Score)
predicting Adverse Clinical Event / ICU Mortality Risk.
"""

import math
import random
from typing import Dict, List, Tuple, Any

FEATURE_NAMES = [
    "gender",
    "age",
    "ethnicity",
    "heart_rate",
    "systolic_bp",
    "glucose",
    "creatinine",
    "oxygen_sat",
    "lactate",
    "comorbidity_score"
]

SENSITIVE_ATTRIBUTES = ["gender", "age_group", "ethnicity"]

def sigmoid(z: float) -> float:
    z = max(-20.0, min(20.0, z))
    return 1.0 / (1.0 + math.exp(-z))

def generate_patient_record(
    gender: int,         # 0: Female, 1: Male
    age_group: int,      # 0: <60, 1: >=60
    ethnicity: int,      # 0: Group A, 1: Group B, 2: Group C
    hospital_id: str,
    base_bias_shift: float = 0.0,
    rng: random.Random = None
) -> Dict[str, Any]:
    if rng is None:
        rng = random.Random()

    # Clinical features generation with realistic healthcare ranges
    age = rng.randint(20, 58) if age_group == 0 else rng.randint(60, 88)
    heart_rate = rng.gauss(82, 14)
    systolic_bp = rng.gauss(128, 18)
    glucose = rng.gauss(115, 32)
    creatinine = max(0.4, rng.gauss(1.1, 0.4))
    oxygen_sat = min(100.0, max(80.0, rng.gauss(96.0, 3.2)))
    lactate = max(0.5, rng.gauss(1.8, 0.8))
    comorbidity_score = rng.randint(0, 4) if age_group == 0 else rng.randint(1, 6)

    # Standardized normalized features (z-score approximation)
    norm_gender = 1.0 if gender == 1 else -1.0
    norm_age = (age - 55.0) / 18.0
    norm_ethnicity = (ethnicity - 1.0)
    norm_hr = (heart_rate - 80.0) / 15.0
    norm_sbp = (systolic_bp - 125.0) / 20.0
    norm_glucose = (glucose - 110.0) / 35.0
    norm_creatinine = (creatinine - 1.0) / 0.5
    norm_o2 = (95.0 - oxygen_sat) / 4.0
    norm_lactate = (lactate - 1.5) / 0.8
    norm_comorb = (comorbidity_score - 2.5) / 1.5

    features = [
        norm_gender,
        norm_age,
        norm_ethnicity,
        norm_hr,
        norm_sbp,
        norm_glucose,
        norm_creatinine,
        norm_o2,
        norm_lactate,
        norm_comorb
    ]

    # Ground truth clinical risk function
    # Biological/Clinical risk + historical systemic disparities (biases to detect and mitigate)
    logit = (
        -0.8
        + 0.55 * norm_age
        + 0.35 * norm_hr
        + 0.40 * norm_sbp
        + 0.30 * norm_glucose
        + 0.50 * norm_creatinine
        + 0.60 * norm_o2
        + 0.65 * norm_lactate
        + 0.45 * norm_comorb
    )

    # Injected demographic bias representing historical disparities / unequal diagnosis
    # Female patients (gender=0) or minority groups historically facing diagnostic disparity or altered thresholds
    if hospital_id == "hospital_b":  # Rural clinic with historical age & resource bias
        logit += 0.30 * age_group - 0.25 * (1 - gender) + base_bias_shift
    elif hospital_id == "hospital_c":  # Cardiac institute with male-centric diagnostic threshold
        logit += 0.40 * gender + base_bias_shift
    else:  # Hospital A (urban center)
        logit += base_bias_shift

    prob = sigmoid(logit)
    label = 1 if rng.random() < prob else 0

    return {
        "features": features,
        "label": label,
        "gender": gender,          # 0: Female, 1: Male
        "age_group": age_group,    # 0: Young (<60), 1: Senior (>=60)
        "ethnicity": ethnicity,    # 0: Group A, 1: Group B, 2: Group C
        "raw_data": {
            "age": age,
            "heart_rate": round(heart_rate, 1),
            "systolic_bp": round(systolic_bp, 1),
            "glucose": round(glucose, 1),
            "creatinine": round(creatinine, 2),
            "oxygen_sat": round(oxygen_sat, 1),
            "lactate": round(lactate, 2),
            "comorbidity_score": comorbidity_score
        }
    }

def generate_federated_healthcare_datasets(seed: int = 42) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """
    Generates non-IID hospital client datasets and a multi-center global test cohort.
    - Hospital A (Urban Medical Center): 1,200 records (balanced demographics)
    - Hospital B (Rural Community Hospital): 800 records (skewed older demographic, sample bias)
    - Hospital C (Specialty Heart Institute): 1,000 records (skewed male cardiac admissions)
    - Global Test Cohort: 1,000 records (representative multi-center distribution)
    """
    rng = random.Random(seed)
    hospitals = {}

    # Hospital A: Urban Medical Center (Balanced, 1200 records)
    hosp_a_data = []
    for _ in range(1200):
        g = 1 if rng.random() < 0.50 else 0
        a = 1 if rng.random() < 0.50 else 0
        eth_roll = rng.random()
        eth = 0 if eth_roll < 0.55 else (1 if eth_roll < 0.85 else 2)
        rec = generate_patient_record(g, a, eth, "hospital_a", rng=rng)
        hosp_a_data.append(rec)
    hospitals["hospital_a"] = hosp_a_data

    # Hospital B: Rural Community Hospital (Older, Under-resourced, 800 records)
    hosp_b_data = []
    for _ in range(800):
        g = 1 if rng.random() < 0.48 else 0
        a = 1 if rng.random() < 0.72 else 0  # 72% Senior population
        eth_roll = rng.random()
        eth = 0 if eth_roll < 0.70 else (1 if eth_roll < 0.90 else 2)
        rec = generate_patient_record(g, a, eth, "hospital_b", base_bias_shift=0.15, rng=rng)
        hosp_b_data.append(rec)
    hospitals["hospital_b"] = hosp_b_data

    # Hospital C: Specialty Heart & Vascular Institute (Male skewed, 1000 records)
    hosp_c_data = []
    for _ in range(1000):
        g = 1 if rng.random() < 0.75 else 0  # 75% Male admissions
        a = 1 if rng.random() < 0.55 else 0
        eth_roll = rng.random()
        eth = 0 if eth_roll < 0.40 else (1 if eth_roll < 0.75 else 2)
        rec = generate_patient_record(g, a, eth, "hospital_c", rng=rng)
        hosp_c_data.append(rec)
    hospitals["hospital_c"] = hosp_c_data

    # Global Test Cohort (1000 patients across mixed centers)
    global_test = []
    for i in range(1000):
        h_type = "hospital_a" if i % 3 == 0 else ("hospital_b" if i % 3 == 1 else "hospital_c")
        g = 1 if rng.random() < 0.50 else 0
        a = 1 if rng.random() < 0.50 else 0
        eth_roll = rng.random()
        eth = 0 if eth_roll < 0.50 else (1 if eth_roll < 0.80 else 2)
        rec = generate_patient_record(g, a, eth, h_type, rng=rng)
        global_test.append(rec)

    return hospitals, global_test

def load_hospital_dataset_from_csv(csv_path: str) -> List[Dict[str, Any]]:
    """
    Loads custom hospital patient data from a CSV file.
    Expected columns (case-insensitive):
      - gender (0/1 or Female/Male or F/M)
      - age (integer/float)
      - ethnicity (optional: 0, 1, 2 or string)
      - heart_rate, systolic_bp, glucose, creatinine, oxygen_sat, lactate, comorbidity_score (clinical vitals)
      - label (0 or 1: e.g. normal vs adverse event/mortality)
    """
    import csv

    dataset = []
    with open(csv_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Lowercase keys for flexible matching
            clean_row = {k.strip().lower(): v.strip() for k, v in row.items() if k}

            # Gender parsing
            g_raw = str(clean_row.get("gender", "0")).lower()
            gender = 0 if g_raw in ["0", "f", "female", "w", "woman"] else 1

            # Age & age group
            age = float(clean_row.get("age", 50))
            age_group = 1 if age >= 60 else 0

            # Ethnicity parsing
            eth_raw = str(clean_row.get("ethnicity", "0")).lower()
            if eth_raw in ["0", "a", "alpha", "group a"]:
                ethnicity = 0
            elif eth_raw in ["1", "b", "beta", "group b"]:
                ethnicity = 1
            elif eth_raw in ["2", "c", "gamma", "group c"]:
                ethnicity = 2
            else:
                try:
                    ethnicity = int(eth_raw) % 3
                except ValueError:
                    ethnicity = 0

            # Vitals
            heart_rate = float(clean_row.get("heart_rate", 80))
            systolic_bp = float(clean_row.get("systolic_bp", 120))
            glucose = float(clean_row.get("glucose", 110))
            creatinine = float(clean_row.get("creatinine", 1.0))
            oxygen_sat = float(clean_row.get("oxygen_sat", 96.0))
            lactate = float(clean_row.get("lactate", 1.5))
            comorbidity_score = float(clean_row.get("comorbidity_score", 1.0))

            # Target label (0 or 1)
            lbl_raw = clean_row.get("label", clean_row.get("target", clean_row.get("outcome", "0")))
            label = 1 if str(lbl_raw) in ["1", "true", "yes", "positive", "high"] else 0

            # Standardized normalized features (z-score)
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

            features = [
                norm_gender,
                norm_age,
                norm_ethnicity,
                norm_hr,
                norm_sbp,
                norm_glucose,
                norm_creatinine,
                norm_o2,
                norm_lactate,
                norm_comorb
            ]

            dataset.append({
                "features": features,
                "label": label,
                "gender": gender,
                "age_group": age_group,
                "ethnicity": ethnicity,
                "raw_data": {
                    "age": int(age),
                    "heart_rate": round(heart_rate, 1),
                    "systolic_bp": round(systolic_bp, 1),
                    "glucose": round(glucose, 1),
                    "creatinine": round(creatinine, 2),
                    "oxygen_sat": round(oxygen_sat, 1),
                    "lactate": round(lactate, 2),
                    "comorbidity_score": int(comorbidity_score)
                }
            })

    return dataset

def export_sample_csv(dataset: List[Dict[str, Any]], filepath: str):
    """Exports dataset records to standard CSV format."""
    import csv
    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "gender", "age", "ethnicity", "heart_rate", "systolic_bp",
            "glucose", "creatinine", "oxygen_sat", "lactate", "comorbidity_score", "label"
        ])
        for d in dataset:
            v = d["raw_data"]
            g_str = "Female" if d["gender"] == 0 else "Male"
            eth_str = f"Group_{chr(65 + d['ethnicity'])}"
            writer.writerow([
                g_str, v["age"], eth_str, v["heart_rate"], v["systolic_bp"],
                v["glucose"], v["creatinine"], v["oxygen_sat"], v["lactate"], v["comorbidity_score"], d["label"]
            ])

BENCHMARK_DATASETS_META = {
    "mimic_iv": {
        "name": "MIMIC-IV (PhysioNet)",
        "type": "ICU Electronic Health Records (EHR)",
        "records_approx": "Approx. 380,000 patients",
        "url": "https://physionet.org/content/mimiciv",
        "description": "Critical care ICU database with high-acuity lab measurements, vital signs, and mortality outcomes from Beth Israel Deaconess Medical Center.",
        "bias_profile": "Historically lower representation of minority demographics and elderly females requiring fairness adjustment."
    },
    "eicu": {
        "name": "eICU Collaborative Research Database",
        "type": "Multi-center ICU data (EHR)",
        "records_approx": "Approx. 200,000 patient unit stays",
        "url": "https://physionet.org/content/eicu-crd",
        "description": "Multi-center critical care database gathered from hundreds of hospitals across the US, exhibiting acute institutional non-IID covariate shifts.",
        "bias_profile": "Inter-hospital resource disparities and variations in monitoring protocols."
    },
    "medmnist": {
        "name": "MedMNIST",
        "type": "Medical Imaging (lightweight benchmark)",
        "records_approx": "Over 700,000 images across 12 datasets",
        "url": "https://medmnist.com/",
        "description": "Lightweight 28x28 medical image classification benchmark across multiple modalities (ChestMNIST, BloodMNIST, PathMNIST).",
        "bias_profile": "Modality-specific class imbalance and demographic reporting gaps."
    },
    "covid19_imaging": {
        "name": "COVID-19 Imaging datasets",
        "type": "Chest X-rays and CT scans",
        "records_approx": "Varies (~50,000+ images across sources)",
        "url": "https://www.kaggle.com/andrewmvd/covid19-x-ray-dataset",
        "description": "Global radiological chest radiograph and computed tomography cohort for pulmonary viral pneumonia risk scoring.",
        "bias_profile": "Severe geographical site bias, device manufacturer artifacts, and skewed age distributions."
    },
    "tcga": {
        "name": "TCGA (The Cancer Genome Atlas)",
        "type": "Genomics (DNA, RNA sequencing)",
        "records_approx": "Over 11,000 patients across 33 cancer types",
        "url": "https://www.cancer.gov/ccg/research/genome-sequencing/tcga",
        "description": "Landmark cancer genomics database profiling primary tumor molecular characteristics, mutation rates, and clinical progression.",
        "bias_profile": "Ancestral and gender genomic disparity across distinct cancer types (e.g. BRCA vs LUAD)."
    }
}

def generate_benchmark_dataset_records(dataset_key: str, count: int = 100, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Generates realistic, scientifically aligned cohorts mimicking the statistical 
    distributions of the top 5 medical AI research benchmarks:
    - MIMIC-IV: High ICU lactate & creatinine, senior skew
    - eICU: Multi-center variance, broad vitals spread
    - MedMNIST: Standardized imaging feature proxies
    - COVID-19 Imaging: Severe hypoxia (low oxygen_sat), elevated inflammatory lactate
    - TCGA: High comorbidity, genomic risk markers mapped to metabolic vitals
    """
    rng = random.Random(seed)
    dataset = []

    for i in range(count):
        if dataset_key == "mimic_iv":
            # ICU EHR: Skewed higher age, severe vitals
            g = 1 if rng.random() < 0.54 else 0
            a = 1 if rng.random() < 0.65 else 0
            eth = rng.choice([0, 0, 1, 2])
            age = rng.randint(45, 88) if a == 1 else rng.randint(22, 59)
            hr = rng.gauss(88, 16)
            sbp = rng.gauss(118, 22)
            glucose = rng.gauss(135, 45)
            creatinine = max(0.5, rng.gauss(1.6, 0.7))
            o2 = min(100.0, max(75.0, rng.gauss(93.5, 4.0)))
            lactate = max(0.7, rng.gauss(2.4, 1.1))
            comorb = rng.randint(2, 6)
            shift = 0.20

        elif dataset_key == "eicu":
            # Multi-center EHR: Diverse clinics
            g = 1 if rng.random() < 0.51 else 0
            a = 1 if rng.random() < 0.58 else 0
            eth = rng.choice([0, 1, 2])
            age = rng.randint(35, 85)
            hr = rng.gauss(84, 14)
            sbp = rng.gauss(126, 18)
            glucose = rng.gauss(120, 35)
            creatinine = max(0.4, rng.gauss(1.2, 0.5))
            o2 = min(100.0, max(82.0, rng.gauss(95.5, 3.0)))
            lactate = max(0.5, rng.gauss(1.8, 0.8))
            comorb = rng.randint(1, 5)
            shift = 0.10

        elif dataset_key == "medmnist":
            # Medical Imaging proxy features (standardized distribution)
            g = 1 if rng.random() < 0.50 else 0
            a = 1 if rng.random() < 0.45 else 0
            eth = rng.choice([0, 1, 2])
            age = rng.randint(20, 75)
            hr = rng.gauss(78, 12)
            sbp = rng.gauss(122, 15)
            glucose = rng.gauss(108, 25)
            creatinine = max(0.4, rng.gauss(1.0, 0.3))
            o2 = min(100.0, max(88.0, rng.gauss(97.2, 2.0)))
            lactate = max(0.4, rng.gauss(1.3, 0.5))
            comorb = rng.randint(0, 3)
            shift = -0.10

        elif dataset_key == "covid19_imaging":
            # COVID-19 Pulmonary Cohort: Low O2 Saturation, high lactate, inflammatory markers
            g = 1 if rng.random() < 0.58 else 0
            a = 1 if rng.random() < 0.62 else 0
            eth = rng.choice([0, 1, 1, 2])
            age = rng.randint(30, 86)
            hr = rng.gauss(96, 18)
            sbp = rng.gauss(132, 20)
            glucose = rng.gauss(128, 38)
            creatinine = max(0.6, rng.gauss(1.4, 0.6))
            o2 = min(98.0, max(72.0, rng.gauss(88.5, 5.5))) # Acute Hypoxemia
            lactate = max(1.0, rng.gauss(2.8, 1.2))
            comorb = rng.randint(2, 6)
            shift = 0.35

        elif dataset_key == "tcga":
            # Genomic Oncological Cohort: High comorbidity, targeted tissue pathology
            g = 1 if rng.random() < 0.48 else 0
            a = 1 if rng.random() < 0.70 else 0
            eth = rng.choice([0, 0, 1, 2])
            age = rng.randint(40, 89)
            hr = rng.gauss(82, 13)
            sbp = rng.gauss(125, 16)
            glucose = rng.gauss(116, 30)
            creatinine = max(0.5, rng.gauss(1.3, 0.4))
            o2 = min(100.0, max(85.0, rng.gauss(96.0, 2.5)))
            lactate = max(0.6, rng.gauss(2.1, 0.9))
            comorb = rng.randint(3, 6) # High chronic morbidity
            shift = 0.25

        else:
            # Standard generic clinic fallback
            g = 1 if rng.random() < 0.50 else 0
            a = 1 if rng.random() < 0.50 else 0
            eth = rng.choice([0, 1, 2])
            age = rng.randint(25, 75)
            hr = rng.gauss(80, 15)
            sbp = rng.gauss(125, 18)
            glucose = rng.gauss(110, 30)
            creatinine = max(0.4, rng.gauss(1.1, 0.4))
            o2 = min(100.0, max(85.0, rng.gauss(96.0, 3.0)))
            lactate = max(0.5, rng.gauss(1.5, 0.7))
            comorb = rng.randint(0, 4)
            shift = 0.0

        # Feature normalization
        norm_gender = 1.0 if g == 1 else -1.0
        norm_age = (age - 55.0) / 18.0
        norm_ethnicity = float(eth - 1.0)
        norm_hr = (hr - 80.0) / 15.0
        norm_sbp = (sbp - 125.0) / 20.0
        norm_glucose = (glucose - 110.0) / 35.0
        norm_creatinine = (creatinine - 1.0) / 0.5
        norm_o2 = (95.0 - o2) / 4.0
        norm_lactate = (lactate - 1.5) / 0.8
        norm_comorb = (comorb - 2.5) / 1.5

        features = [
            norm_gender, norm_age, norm_ethnicity, norm_hr, norm_sbp,
            norm_glucose, norm_creatinine, norm_o2, norm_lactate, norm_comorb
        ]

        logit = (
            -0.7
            + 0.50 * norm_age
            + 0.35 * norm_hr
            + 0.40 * norm_sbp
            + 0.30 * norm_glucose
            + 0.50 * norm_creatinine
            + 0.65 * norm_o2
            + 0.70 * norm_lactate
            + 0.45 * norm_comorb
            + shift
        )
        prob = sigmoid(logit)
        label = 1 if rng.random() < prob else 0

        dataset.append({
            "features": features,
            "label": label,
            "gender": g,
            "age_group": 1 if age >= 60 else 0,
            "ethnicity": eth,
            "raw_data": {
                "age": int(age),
                "heart_rate": round(hr, 1),
                "systolic_bp": round(sbp, 1),
                "glucose": round(glucose, 1),
                "creatinine": round(creatinine, 2),
                "oxygen_sat": round(o2, 1),
                "lactate": round(lactate, 2),
                "comorbidity_score": int(comorb)
            }
        })

    return dataset

def generate_benchmark_csv_text(dataset_key: str, count: int = 100, seed: int = 42) -> str:
    """Exports benchmark records to CSV format with header comments and clean rows."""
    import io, csv
    dataset = generate_benchmark_dataset_records(dataset_key, count=count, seed=seed)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "gender", "age", "ethnicity", "heart_rate", "systolic_bp",
        "glucose", "creatinine", "oxygen_sat", "lactate", "comorbidity_score", "label"
    ])
    for d in dataset:
        v = d["raw_data"]
        g_str = "Female" if d["gender"] == 0 else "Male"
        eth_str = f"Group_{chr(65 + d['ethnicity'])}"
        writer.writerow([
            g_str, v["age"], eth_str, v["heart_rate"], v["systolic_bp"],
            v["glucose"], v["creatinine"], v["oxygen_sat"], v["lactate"], v["comorbidity_score"], d["label"]
        ])
    return output.getvalue()



