"""
Kaggle / UCI Diabetes 130-US Hospitals (1999-2008) Dataset Ingestion & Feature Engineering Engine
Dataset: Humberto Brandao (Kaggle: brandao/diabetes) / UCI ML Repository
ABESEC Ghaziabad · Session 2023-24
Project Team: Sneha, Tanisha, Suhani Saraswat | Guide: Dr. Anshul Tickoo
"""

import os
import csv
import json
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CSV_PATH = os.path.join(DATA_DIR, "diabetic_data.csv")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Age bracket mapping to midpoint
AGE_MAP = {
    "[0-10)": 5,
    "[10-20)": 15,
    "[20-30)": 25,
    "[30-40)": 35,
    "[40-50)": 45,
    "[50-60)": 55,
    "[60-70)": 65,
    "[70-80)": 75,
    "[80-90)": 85,
    "[90-100)": 95
}

def is_icd(code_val: str, prefixes: list) -> bool:
    if not code_val or code_val == "?":
        return False
    return any(code_val.startswith(p) for p in prefixes)

def load_and_process_kaggle_dataset(max_samples=None, random_seed=42):
    """
    Parses diabetic_data.csv and extracts:
    - 15-feature matrix X matching MediPersonal AI clinical vector
    - Target treatment recommendation label y (0 to 5)
    - Outcome readmission label y_readmit
    - Metadata and statistical distributions
    """
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"diabetic_data.csv not found at {CSV_PATH}. Run download_dataset.py first.")

    np.random.seed(random_seed)
    
    X_rows = []
    y_targets = []
    y_readmit = []
    curated_cases = []
    
    total_encounters = 0
    unique_patients = set()
    gender_counts = {"Female": 0, "Male": 0}
    age_dist = {}
    med_dist = {
        "Metformin": 0,
        "Insulin": 0,
        "Sulfonylurea": 0,
        "DPP4_or_TZD": 0,
        "Combination": 0
    }
    readmit_dist = {"NO": 0, ">30": 0, "<30": 0}
    
    feature_names = [
        "age", "gender", "condition", "disease_duration", "egfr", "hba1c", "systolic_bp",
        "comorbidity_ckd", "comorbidity_chf", "comorbidity_asthma", "prev_response",
        "allergy_sulfa", "allergy_penicillin", "allergy_acei", "prev_adverse"
    ]

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            gender_raw = row.get("gender", "")
            if gender_raw not in ["Female", "Male"]:
                continue
                
            total_encounters += 1
            unique_patients.add(row.get("patient_nbr"))
            
            # 1. Demographics
            age = AGE_MAP.get(row.get("age"), 55)
            gender = 1.0 if gender_raw == "Male" else 0.0
            gender_counts[gender_raw] += 1
            age_bucket = row.get("age", "[50-60)")
            age_dist[age_bucket] = age_dist.get(age_bucket, 0) + 1
            
            # Diagnoses
            d1 = str(row.get("diag_1", ""))
            d2 = str(row.get("diag_2", ""))
            d3 = str(row.get("diag_3", ""))
            all_diags = [d1, d2, d3]
            
            # Comorbidities
            is_ckd = any(is_icd(d, ["580", "581", "582", "583", "584", "585", "586", "587", "588", "589"]) for d in all_diags)
            is_chf = any(is_icd(d, ["428", "390", "391", "392", "402", "404", "410", "411", "412", "413", "414", "425"]) for d in all_diags)
            is_asthma = any(is_icd(d, ["493", "490", "491", "492", "496"]) for d in all_diags)
            is_htn = any(is_icd(d, ["401", "402", "403", "404", "405"]) for d in all_diags)
            
            # Primary condition (0: Diabetes, 1: Hypertension, 2: Asthma, 3: CVD)
            if is_icd(d1, ["250"]) or is_icd(d2, ["250"]):
                condition = 0.0
            elif is_htn and is_icd(d1, ["401", "402", "403", "404", "405"]):
                condition = 1.0
            elif is_asthma and is_icd(d1, ["493"]):
                condition = 2.0
            elif is_chf and is_icd(d1, ["39", "40", "41", "42", "43", "44", "45"]):
                condition = 3.0
            else:
                condition = 0.0
                
            # History & Utilization
            num_inpatient = float(row.get("number_inpatient", 0) or 0)
            num_outpatient = float(row.get("number_outpatient", 0) or 0)
            num_emergency = float(row.get("number_emergency", 0) or 0)
            num_meds = float(row.get("num_medications", 10) or 10)
            time_in_hosp = float(row.get("time_in_hospital", 3) or 3)
            num_diags = float(row.get("number_diagnoses", 5) or 5)
            
            # Estimate disease duration from chronic burden & past visits
            disease_duration = round(min(22.0, max(0.5, 2.0 + (num_diags * 0.7) + (num_inpatient * 1.5) + (time_in_hosp * 0.4))), 1)
            
            # Estimate renal eGFR based on CKD code, age, gender, and encounter severity
            if is_ckd:
                egfr = round(float(np.random.uniform(22.0, 48.0)), 1)
            else:
                base_egfr = 105.0 - (age * 0.45)
                if gender == 0.0:
                    base_egfr *= 0.92
                egfr = round(float(np.clip(base_egfr + np.random.normal(0, 8), 45.0, 118.0)), 1)
                
            # Glycemic control / HbA1c from A1Cresult
            a1c_raw = row.get("A1Cresult", "None")
            if a1c_raw == ">8":
                hba1c = round(float(np.random.uniform(8.8, 12.4)), 1)
            elif a1c_raw == ">7":
                hba1c = round(float(np.random.uniform(7.4, 8.6)), 1)
            elif a1c_raw == "Norm":
                hba1c = round(float(np.random.uniform(5.2, 6.4)), 1)
            else:
                # None recorded; correlate with insulin use and time in hospital
                if row.get("insulin") in ["Up", "Down", "Steady"]:
                    hba1c = round(float(np.random.uniform(7.8, 10.2)), 1)
                else:
                    hba1c = round(float(np.random.uniform(6.5, 8.4)), 1)
                    
            # Systolic BP
            if is_htn:
                systolic_bp = round(float(np.clip(140.0 + (num_emergency * 4.0) + np.random.normal(0, 10), 128.0, 188.0)), 0)
            else:
                systolic_bp = round(float(np.clip(118.0 + (age * 0.15) + np.random.normal(0, 8), 102.0, 142.0)), 0)
                
            # Readmission & Past response
            readmit_raw = row.get("readmitted", "NO")
            readmit_dist[readmit_raw] = readmit_dist.get(readmit_raw, 0) + 1
            change_raw = row.get("change", "No")
            
            if readmit_raw == "<30":
                prev_resp = 0.0  # Adverse / Severe relapse
                prev_adv = 1.0
                readmit_code = 2
            elif readmit_raw == ">30":
                prev_resp = 2.0 if change_raw == "No" else 1.0  # Moderate / Poor
                prev_adv = 1.0 if num_emergency > 1 else 0.0
                readmit_code = 1
            else:
                prev_resp = 3.0  # Good response / Discharge home
                prev_adv = 0.0
                readmit_code = 0
                
            # Allergies (based on realistic clinical intolerance distribution)
            allergy_sulfa = 1.0 if (hash(row.get("encounter_id")) % 11 == 0) else 0.0
            allergy_penicillin = 1.0 if (hash(row.get("patient_nbr")) % 13 == 0) else 0.0
            allergy_acei = 1.0 if (is_htn and (hash(row.get("encounter_id")) % 17 == 0)) else 0.0
            
            # Map prescribed medications to CDSS target classes:
            # 0: Metformin (First-line Biguanide)
            # 1: SGLT2 Inhibitor (Empagliflozin)
            # 2: GLP-1 Receptor Agonist (Semaglutide)
            # 3: DPP-4 Inhibitor (Sitagliptin)
            # 4: Sulfonylurea (Glimepiride)
            # 5: Basal Insulin Glargine
            has_insulin = row.get("insulin") in ["Steady", "Up", "Down"]
            has_metformin = row.get("metformin") in ["Steady", "Up", "Down"]
            has_sulfa = any(row.get(k) in ["Steady", "Up", "Down"] for k in ["glimepiride", "glipizide", "glyburide"])
            has_tzd = any(row.get(k) in ["Steady", "Up", "Down"] for k in ["pioglitazone", "rosiglitazone"])
            
            if has_insulin:
                if has_metformin and egfr >= 45:
                    target = 0 if hba1c < 8.5 else 5
                elif is_ckd or egfr < 30:
                    target = 5  # Insulin is standard of care when renal filtration fails
                else:
                    target = 5  # Insulin
                med_dist["Insulin"] += 1
            elif has_metformin:
                if is_ckd or egfr < 30:
                    # Metformin contraindicated in severe CKD -> shift to renal safe DPP-4
                    target = 3
                elif is_chf:
                    target = 1  # SGLT2i preferred in CHF
                else:
                    target = 0  # Metformin
                med_dist["Metformin"] += 1
            elif has_sulfa:
                if allergy_sulfa == 1.0 or prev_adv == 1.0:
                    target = 2 if hba1c > 8.0 else 3
                else:
                    target = 4  # Sulfonylurea
                med_dist["Sulfonylurea"] += 1
            elif has_tzd:
                target = 3  # DPP-4 / Oral oral alternative
                med_dist["DPP4_or_TZD"] += 1
            else:
                # Diet / other -> guidelines assign based on renal/glycemic status
                if egfr < 30:
                    target = 3  # DPP-4
                elif is_ckd or is_chf:
                    target = 1  # SGLT2i
                elif hba1c > 9.0:
                    target = 5  # Insulin
                else:
                    target = 0  # Metformin
                med_dist["Combination"] += 1

            feature_vec = [
                age, gender, condition, disease_duration, egfr, hba1c, systolic_bp,
                1.0 if is_ckd else 0.0,
                1.0 if is_chf else 0.0,
                1.0 if is_asthma else 0.0,
                prev_resp, allergy_sulfa, allergy_penicillin, allergy_acei, prev_adv
            ]
            
            X_rows.append(feature_vec)
            y_targets.append(target)
            y_readmit.append(readmit_code)
            
            # Save representative real encounter samples for the UI case browser
            if len(curated_cases) < 120 and (i % 800 == 0 or (is_ckd and len(curated_cases) < 60)):
                curated_cases.append({
                    "encounter_id": row.get("encounter_id"),
                    "patient_nbr": row.get("patient_nbr"),
                    "age": age_bucket,
                    "age_num": age,
                    "gender": gender_raw,
                    "time_in_hospital_days": time_in_hosp,
                    "num_medications": int(num_meds),
                    "num_lab_procedures": int(row.get("num_lab_procedures", 0) or 0),
                    "primary_diagnosis": d1,
                    "secondary_diagnosis": d2,
                    "tertiary_diagnosis": d3,
                    "a1c_result": a1c_raw,
                    "hba1c_estimated": hba1c,
                    "egfr_estimated": egfr,
                    "systolic_bp": int(systolic_bp),
                    "comorbidities": [c for c, flag in [("CKD", is_ckd), ("CHF/CVD", is_chf), ("Asthma", is_asthma), ("Hypertension", is_htn)] if flag],
                    "active_hospital_medications": [k for k in ["metformin", "insulin", "glimepiride", "glipizide", "glyburide", "pioglitazone"] if row.get(k) in ["Steady", "Up", "Down"]],
                    "readmitted": readmit_raw,
                    "recommended_target_class": target
                })
                
            if max_samples and len(X_rows) >= max_samples:
                break
                
    X = np.array(X_rows, dtype=float)
    y = np.array(y_targets, dtype=int)
    y_readmit = np.array(y_readmit, dtype=int)
    
    metadata = {
        "dataset_name": "Diabetes 130-US Hospitals (1999-2008)",
        "source": "Humberto Brandao, Ph.D. (Kaggle) / UCI Machine Learning Repository",
        "kaggle_url": "https://www.kaggle.com/datasets/brandao/diabetes/data",
        "total_records": total_encounters,
        "unique_patients": len(unique_patients),
        "hospitals_represented": 130,
        "time_period": "1999 - 2008 (10 Years)",
        "features_processed": len(feature_names),
        "feature_names": feature_names,
        "gender_distribution": gender_counts,
        "age_distribution": age_dist,
        "medication_distribution": med_dist,
        "readmission_distribution": readmit_dist,
        "target_classes": [
            "Metformin (First-line Biguanide)",
            "SGLT2 Inhibitor (Empagliflozin)",
            "GLP-1 Receptor Agonist (Semaglutide)",
            "DPP-4 Inhibitor (Sitagliptin)",
            "Sulfonylurea (Glimepiride)",
            "Basal Insulin Glargine"
        ]
    }
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "dataset_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    with open(os.path.join(MODELS_DIR, "kaggle_patient_cohort.json"), "w") as f:
        json.dump(curated_cases, f, indent=2)
        
    print(f"Processed {len(X)} records from Kaggle 130-US Hospitals dataset.")
    print(f"Unique patients: {len(unique_patients)}, Metadata saved to {MODELS_DIR}")
    return X, y, y_readmit, feature_names, metadata

if __name__ == "__main__":
    X, y, y_readmit, feature_names, metadata = load_and_process_kaggle_dataset()
    print("Shape:", X.shape, y.shape)
    print("Class counts:", np.bincount(y))
