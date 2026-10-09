"""
Synthetic EHR Dataset Generator for Personalized Treatment Recommendation
ABESEC Ghaziabad - Session 2023-24
Group: Sneha, Tanisha, Suhani Saraswat | Guide: Dr. Anshul Tickoo
"""

import numpy as np
import json

CONDITIONS = ["Type 2 Diabetes", "Hypertension", "Asthma", "Cardiovascular Disease"]

DIABETES_TREATMENTS = [
    "Metformin (First-line Biguanide)",
    "SGLT2 Inhibitor (Empagliflozin)",
    "GLP-1 Receptor Agonist (Semaglutide)",
    "DPP-4 Inhibitor (Sitagliptin)",
    "Sulfonylurea (Glimepiride)",
    "Basal Insulin Glargine"
]

HYPERTENSION_TREATMENTS = [
    "ACE Inhibitor (Lisinopril)",
    "ARB (Losartan)",
    "Calcium Channel Blocker (Amlodipine)",
    "Thiazide Diuretic (Chlorthalidone)",
    "Beta-Blocker (Metoprolol Succinate)"
]

ASTHMA_TREATMENTS = [
    "Inhaled Corticosteroid + LABA (Budesonide/Formoterol)",
    "Leukotriene Receptor Antagonist (Montelukast)",
    "Short-Acting Beta Agonist (Albuterol)",
    "Long-Acting Muscarinic Antagonist (Tiotropium)",
    "Biologic Anti-IgE (Omalizumab)"
]

CVD_TREATMENTS = [
    "High-Intensity Statin (Atorvastatin 40mg)",
    "Antiplatelet (Aspirin 81mg)",
    "Dual Antiplatelet (Aspirin + Clopidogrel)",
    "Beta-Blocker (Carvedilol)",
    "PCSK9 Inhibitor (Evolocumab)"
]

def generate_patient_dataset(n_samples=1200, random_state=42):
    """
    Generates a realistic clinical dataset reflecting multi-feature patient history.
    Features:
      - age (18 - 85)
      - gender (0: Female, 1: Male)
      - condition (0: Diabetes, 1: Hypertension, 2: Asthma, 3: CVD)
      - disease_duration_years (0.5 - 25)
      - egfr (15 - 120 mL/min/1.73m²)
      - hba1c (5.4 - 12.5 %)
      - systolic_bp (95 - 195 mmHg)
      - comorbidity_ckd (0/1)
      - comorbidity_chf (0/1)
      - comorbidity_asthma (0/1)
      - prev_response (0: Adverse, 1: Poor, 2: Moderate, 3: Good)
      - allergy_sulfa (0/1)
      - allergy_penicillin (0/1)
      - allergy_acei (0/1)
      - prev_adverse_reaction (0/1)
    """
    np.random.seed(random_state)
    
    age = np.random.randint(22, 82, size=n_samples)
    gender = np.random.choice([0, 1], size=n_samples)
    condition = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.4, 0.35, 0.15, 0.10])
    disease_duration = np.round(np.random.uniform(0.5, 20.0, size=n_samples), 1)
    
    # Biomarkers with realistic clinical correlations
    egfr = np.clip(np.random.normal(78 - (age - 40) * 0.45, 18, size=n_samples), 18, 120).round(1)
    hba1c = np.where(condition == 0, np.clip(np.random.normal(8.4, 1.4, size=n_samples), 6.0, 13.0), np.clip(np.random.normal(5.7, 0.5, size=n_samples), 5.0, 6.8)).round(1)
    systolic_bp = np.where(condition == 1, np.clip(np.random.normal(148, 16, size=n_samples), 115, 195), np.clip(np.random.normal(124, 12, size=n_samples), 100, 150)).round(0)
    
    # Comorbidities
    ckd = (egfr < 60).astype(int)
    chf = np.random.choice([0, 1], size=n_samples, p=[0.82, 0.18])
    asthma_comorb = np.where(condition == 2, 1, np.random.choice([0, 1], size=n_samples, p=[0.92, 0.08]))
    
    # Allergies & History
    allergy_sulfa = np.random.choice([0, 1], size=n_samples, p=[0.90, 0.10])
    allergy_penicillin = np.random.choice([0, 1], size=n_samples, p=[0.88, 0.12])
    allergy_acei = np.random.choice([0, 1], size=n_samples, p=[0.93, 0.07])
    prev_adverse = np.random.choice([0, 1], size=n_samples, p=[0.83, 0.17])
    prev_response = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.12, 0.23, 0.35, 0.30])
    
    # Treatment target assignment logic based on evidence-based medical guidelines (ADA, ACC/AHA, GINA)
    targets = []
    for i in range(n_samples):
        cond = condition[i]
        c_egfr = egfr[i]
        c_a1c = hba1c[i]
        c_bp = systolic_bp[i]
        c_ckd = ckd[i]
        c_chf = chf[i]
        c_asthma = asthma_comorb[i]
        c_sulfa = allergy_sulfa[i]
        c_ace_all = allergy_acei[i]
        c_resp = prev_response[i]
        c_adv = prev_adverse[i]
        
        if cond == 0:  # Type 2 Diabetes
            # If eGFR < 30, Metformin is contraindicated -> GLP-1 or DPP-4 or Insulin
            if c_egfr < 30:
                if c_a1c > 9.5:
                    target = 5  # Insulin
                elif c_resp >= 2:
                    target = 3  # DPP-4i (renal safe)
                else:
                    target = 2  # GLP-1 RA
            elif c_ckd == 1 or c_chf == 1:
                # SGLT2i has strong organ protection in CKD/CHF
                target = 1  # SGLT2i
            elif c_sulfa == 1 or c_adv == 1:
                # Avoid Sulfonylurea if sulfa allergic or adverse history
                target = 2 if c_a1c > 8.0 else 0  # GLP-1 RA or Metformin
            elif c_a1c > 9.8:
                target = 5  # Basal Insulin
            else:
                target = 0  # Metformin
                
        elif cond == 1:  # Hypertension
            if c_ckd == 1 or c_chf == 1:
                # ACEi or ARB preferred for renal/cardiac protection, unless allergic
                if c_ace_all == 1:
                    target = 1  # ARB
                else:
                    target = 0  # ACEi
            elif c_bp > 160:
                target = 2  # CCB (potent monotherapy/combo)
            elif c_asthma == 1:
                # Avoid Beta-blocker in asthma
                target = 1 if c_ace_all == 1 else 0  # ARB or ACEi
            elif c_resp <= 1 and c_adv == 1:
                target = 2  # CCB
            else:
                target = 3  # Thiazide
                
        elif cond == 2:  # Asthma
            if c_resp >= 2:
                target = 0  # ICS + LABA
            elif c_resp == 1:
                target = 3  # LAMA add-on
            else:
                target = 4  # Biologic
                
        else:  # CVD
            if c_chf == 1:
                target = 3  # Carvedilol
            elif c_resp >= 2:
                target = 0  # Atorvastatin
            else:
                target = 2  # DAPT
                
        targets.append(target)
        
    X = np.column_stack([
        age, gender, condition, disease_duration, egfr, hba1c, systolic_bp,
        ckd, chf, asthma_comorb, prev_response, allergy_sulfa, allergy_penicillin,
        allergy_acei, prev_adverse
    ])
    y = np.array(targets)
    
    feature_names = [
        "age", "gender", "condition", "disease_duration", "egfr", "hba1c", "systolic_bp",
        "comorbidity_ckd", "comorbidity_chf", "comorbidity_asthma", "prev_response",
        "allergy_sulfa", "allergy_penicillin", "allergy_acei", "prev_adverse"
    ]
    
    return X, y, feature_names
