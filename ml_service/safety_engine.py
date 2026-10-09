"""
Rule-Based Clinical Safety Layer
ABESEC Ghaziabad - Session 2023-24
Group: Sneha, Tanisha, Suhani Saraswat | Guide: Dr. Anshul Tickoo

Implements Step 8 of Synopsis:
1. Patient allergy match
2. Previous adverse reaction check
3. Comorbidity contraindications
4. Known medication interactions
5. Missing critical clinical information alerts
"""

from typing import Dict, List, Any

# Clinical Contraindication Rules
CONTRAINDICATION_RULES = [
    {
        "id": "MET_RENAL_FAIL",
        "treatment_pattern": "Metformin",
        "condition": lambda p: p.get("egfr", 90) < 30 or "end-stage renal" in p.get("comorbidity", "").lower(),
        "severity": "CRITICAL_CONTRAINDICATION",
        "title": "Severe Renal Impairment Contraindication",
        "message": "Metformin is strictly contraindicated when eGFR < 30 mL/min/1.73m² due to significant risk of lactic acidosis.",
        "action": "EXCLUDE"
    },
    {
        "id": "MET_RENAL_WARN",
        "treatment_pattern": "Metformin",
        "condition": lambda p: 30 <= p.get("egfr", 90) < 45,
        "severity": "WARNING",
        "title": "Renal Dose Adjustment Required",
        "message": "Patient eGFR is 30-44 mL/min/1.73m². Maximum recommended dose is 1000 mg/day with frequent renal function monitoring.",
        "action": "DOWNRANK"
    },
    {
        "id": "BB_ASTHMA_BLOCK",
        "treatment_pattern": "Beta-Blocker",
        "condition": lambda p: "asthma" in p.get("condition", "").lower() or "asthma" in p.get("comorbidity", "").lower(),
        "severity": "CRITICAL_CONTRAINDICATION",
        "title": "Bronchospasm Risk in Asthma",
        "message": "Beta-blockers can cause severe bronchospasm and attenuate bronchodilator efficacy in asthmatic patients.",
        "action": "EXCLUDE"
    },
    {
        "id": "ACE_HYPERKALEMIA",
        "treatment_pattern": "ACE Inhibitor",
        "condition": lambda p: p.get("potassium", 4.2) > 5.2 or "hyperkalemia" in p.get("comorbidity", "").lower(),
        "severity": "HIGH_RISK",
        "title": "Hyperkalemia Exacerbation",
        "message": "ACE Inhibitors reduce aldosterone and can precipitate life-threatening hyperkalemia when baseline K+ > 5.2 mEq/L.",
        "action": "DOWNRANK"
    },
    {
        "id": "SULFA_ALLERGY_SU",
        "treatment_pattern": "Sulfonylurea",
        "condition": lambda p: "sulfa" in p.get("allergy", "").lower(),
        "severity": "ALLERGY_MATCH",
        "title": "Sulfa Cross-Reactivity Alert",
        "message": "Patient has documented Sulfa allergy. Sulfonylureas (Glimepiride) contain a sulfonamide moiety with cross-allergy risk.",
        "action": "EXCLUDE"
    },
    {
        "id": "ACEI_ALLERGY",
        "treatment_pattern": "ACE Inhibitor",
        "condition": lambda p: "ace" in p.get("allergy", "").lower() or "angioedema" in p.get("previous_adverse", "").lower(),
        "severity": "ALLERGY_MATCH",
        "title": "ACE-Inhibitor / Angioedema History",
        "message": "Documented ACE-inhibitor allergy or prior angioedema. Exclude ACE inhibitors; consider ARB with extreme caution or alternative class.",
        "action": "EXCLUDE"
    },
    {
        "id": "PENICILLIN_ALERT",
        "treatment_pattern": "Penicillin",
        "condition": lambda p: "penicillin" in p.get("allergy", "").lower(),
        "severity": "ALLERGY_MATCH",
        "title": "Penicillin Anaphylaxis Precaution",
        "message": "Documented Penicillin allergy. Beta-lactam antibiotics must be avoided.",
        "action": "EXCLUDE"
    },
    {
        "id": "SGLT2_DKA_RISK",
        "treatment_pattern": "SGLT2 Inhibitor",
        "condition": lambda p: p.get("egfr", 90) < 20,
        "severity": "HIGH_RISK",
        "title": "Low eGFR Glycemic Inefficacy",
        "message": "SGLT2 inhibitors have diminished glycemic efficacy when eGFR < 20 mL/min/1.73m².",
        "action": "DOWNRANK"
    }
]

# Drug-Drug Interaction Rules
DRUG_INTERACTIONS = [
    {
        "treatment_pattern": "ACE Inhibitor",
        "interacting_med": ["spironolactone", "potassium", "triamterene"],
        "severity": "HIGH_RISK",
        "message": "Concomitant use of ACE Inhibitor and Potassium-sparing diuretics significantly increases risk of severe hyperkalemia."
    },
    {
        "treatment_pattern": "Beta-Blocker",
        "interacting_med": ["verapamil", "diltiazem"],
        "severity": "HIGH_RISK",
        "message": "Combined Beta-Blocker and non-dihydropyridine CCB therapy may cause profound bradycardia or AV nodal block."
    },
    {
        "treatment_pattern": "Metformin",
        "interacting_med": ["radiocontrast", "contrast"],
        "severity": "WARNING",
        "message": "Metformin should be withheld before and 48 hours after iodinated radiocontrast procedures."
    }
]

def evaluate_patient_safety(patient: Dict[str, Any], candidate_treatments: List[str]) -> Dict[str, Any]:
    """
    Evaluates patient history against safety rules for all candidate treatments.
    Returns alerts, excluded treatments, penalized treatments, and verification status.
    """
    alerts = []
    excluded = set()
    penalties = {}
    
    # 1. Missing Critical Information check
    if "egfr" not in patient or patient.get("egfr") is None:
        alerts.append({
            "severity": "WARNING",
            "title": "Missing eGFR (Renal Function)",
            "message": "Renal filtration rate (eGFR) is not recorded. Renal-cleared drugs require dosage confirmation.",
            "target": "General"
        })
    if "allergy" not in patient or not patient.get("allergy"):
        alerts.append({
            "severity": "INFO",
            "title": "Unconfirmed Allergy Status",
            "message": "Allergy history not explicitly verified with patient or records.",
            "target": "General"
        })
        
    # 2. Previous adverse reactions general check
    prev_adverse = str(patient.get("previous_adverse", "")).strip()
    prev_response = str(patient.get("response", "")).strip().lower()
    if prev_adverse and prev_adverse.lower() != "none" or "adverse" in prev_response:
        alerts.append({
            "severity": "HIGH_RISK",
            "title": "Previous Adverse Reaction Flagged",
            "message": f"Patient has prior documented adverse reaction: '{prev_adverse or 'Adverse event on record'}'. Care must be exercised to avoid chemically related agents.",
            "target": patient.get("medication", "Previous agent")
        })

    # 3. Treatment-specific contraindication and allergy checks
    current_meds = str(patient.get("medication", "")).lower()
    
    for treatment in candidate_treatments:
        penalties[treatment] = 1.0
        
        # Check contraindication rules
        for rule in CONTRAINDICATION_RULES:
            if rule["treatment_pattern"].lower() in treatment.lower():
                try:
                    if rule["condition"](patient):
                        alerts.append({
                            "severity": rule["severity"],
                            "title": rule["title"],
                            "message": f"[{treatment}] {rule['message']}",
                            "target": treatment,
                            "action": rule["action"]
                        })
                        if rule["action"] == "EXCLUDE":
                            excluded.add(treatment)
                            penalties[treatment] = 0.0
                        elif rule["action"] == "DOWNRANK":
                            penalties[treatment] = min(penalties[treatment], 0.45)
                except Exception:
                    pass
                    
        # Check Drug-Drug interactions
        for ddi in DRUG_INTERACTIONS:
            if ddi["treatment_pattern"].lower() in treatment.lower():
                for imed in ddi["interacting_med"]:
                    if imed in current_meds:
                        alerts.append({
                            "severity": ddi["severity"],
                            "title": "Potential Drug-Drug Interaction",
                            "message": f"[{treatment} + {imed}] {ddi['message']}",
                            "target": treatment,
                            "action": "DOWNRANK"
                        })
                        penalties[treatment] = min(penalties[treatment], 0.60)
                        
    # 4. Check if candidate matches previously failed drug
    prev_failed = str(patient.get("medication", "")).lower()
    if prev_failed and prev_failed != "none":
        for treatment in candidate_treatments:
            if prev_failed in treatment.lower() and ("poor" in prev_response or "adverse" in prev_response):
                alerts.append({
                    "severity": "WARNING",
                    "title": "Prior Unfavorable Response",
                    "message": f"Patient previously had poor/adverse response to {prev_failed}. Downranking duplicate regimen.",
                    "target": treatment,
                    "action": "DOWNRANK"
                })
                penalties[treatment] = min(penalties[treatment], 0.30)
                
    return {
        "alerts": alerts,
        "excluded_treatments": list(excluded),
        "penalties": penalties,
        "safe_options_count": len(candidate_treatments) - len(excluded),
        "safety_verified": len([a for a in alerts if a["severity"] in ("CRITICAL_CONTRAINDICATION", "ALLERGY_MATCH")]) == 0
    }
