"""
HCC Reference Data
------------------
CMS Medicare Advantage Risk Adjustment Model (CMS-HCC v24/v28)
Real HCC categories, ICD-10 diagnosis codes, and RAF weights.
"""

import pandas as pd

# ── 1. HCC Category Reference ──────────────────────────────────────────────
HCC_CATEGORIES = {
    # HCC_ID : (description, base_RAF_weight)
    8:   ("Metastatic Cancer and Acute Leukemia",         2.659),
    9:   ("Lung and Other Severe Cancers",                1.276),
    10:  ("Lymphoma and Other Cancers",                   0.670),
    11:  ("Colorectal, Bladder, and Other Cancers",       0.307),
    12:  ("Breast, Prostate, and Other Cancers",          0.154),
    17:  ("Diabetes with Acute Complications",            0.318),
    18:  ("Diabetes with Chronic Complications",          0.318),
    19:  ("Diabetes without Complication",                0.105),
    46:  ("Severe Hematological Disorders",               0.951),
    54:  ("Drug/Alcohol Psychosis",                       0.383),
    55:  ("Drug/Alcohol Dependence",                      0.196),
    57:  ("Schizophrenia",                                0.421),
    58:  ("Major Depressive and Bipolar Disorders",       0.212),
    70:  ("Quadriplegia",                                 1.234),
    71:  ("Paraplegia",                                   0.805),
    72:  ("Spinal Cord Disorders/Injuries",               0.470),
    73:  ("Amyotrophic Lateral Sclerosis and Other MND",  1.634),
    74:  ("Cerebral Palsy",                               0.184),
    75:  ("Myasthenia Gravis/LEMS",                       0.670),
    76:  ("Muscular Dystrophy",                           0.670),
    77:  ("Multiple Sclerosis",                           0.670),
    78:  ("Parkinson's and Huntington's Diseases",        0.670),
    79:  ("Seizure Disorders and Convulsions",            0.254),
    80:  ("Coma, Brain Compression/Anoxic Damage",        0.670),
    82:  ("Respirator Dependence/Tracheostomy Status",    1.634),
    83:  ("Respiratory Arrest",                           1.634),
    84:  ("Cardio-Respiratory Failure and Shock",         0.295),
    85:  ("Congestive Heart Failure",                     0.331),
    86:  ("Acute Myocardial Infarction",                  0.199),
    87:  ("Unstable Angina and Other Acute Ischemic HD",  0.199),
    88:  ("Angina Pectoris",                              0.090),
    96:  ("Specified Heart Arrhythmias",                  0.179),
    108: ("Vascular Disease with Complications",          0.539),
    111: ("Chronic Obstructive Pulmonary Disease",        0.328),
    112: ("Fibrosis of Lung and Other Chronic Lung Dis",  0.328),
    114: ("Aspiration and Specified Bacterial Pneumonia", 0.631),
    134: ("Dialysis Status",                              0.428),
    135: ("Acute Renal Failure",                          0.428),
    136: ("Chronic Kidney Disease, Stage 5",              0.289),
    137: ("Chronic Kidney Disease, Severe (Stage 4)",     0.289),
    138: ("Chronic Kidney Disease, Moderate (Stage 3)",   0.069),
    161: ("Chronic Ulcer of Skin, Except Pressure",       0.515),
    189: ("Amputation Status, Lower Limb/Amputation Comp",0.515),
}

# ── 2. ICD-10 → HCC Mapping (subset, representative) ──────────────────────
ICD_TO_HCC = {
    # Diabetes
    "E1165":  18,  "E1140":  18,  "E1151":  18,  "E1152":  18,
    "E1100":  17,  "E1101":  17,
    "E119":   19,  "E1169":  19,

    # Congestive Heart Failure
    "I5020":  85,  "I5021":  85,  "I5022":  85,  "I5023":  85,
    "I5030":  85,  "I5031":  85,  "I5032":  85,  "I5033":  85,
    "I509":   85,

    # Chronic Kidney Disease
    "N184":   137, "N183":   138, "N185":   136, "N186":   134,
    "Z992":   134,

    # COPD
    "J449":   111, "J440":   111, "J441":   111,

    # Cancer
    "C349":   9,   "C3490":  9,   "C3491":  9,
    "C509":   12,  "C61":    12,
    "C189":   11,  "C20":    11,
    "C7900":  8,   "C7951":  8,

    # Schizophrenia / Mental Health
    "F209":   57,  "F200":   57,
    "F319":   58,  "F329":   58,  "F330":   58,

    # Vascular
    "I7000":  108, "I7001":  108, "I702":   108,

    # Arrhythmia
    "I4891":  96,  "I489":   96,  "I4819":  96,

    # Renal Failure
    "N179":   135, "N170":   135,

    # Amputation
    "Z8961":  189, "Z8962":  189,
}

def get_hcc_for_icd(icd_code: str) -> int | None:
    """Return HCC category for a given ICD-10 code, or None if not mapped."""
    return ICD_TO_HCC.get(icd_code.replace(".", "").upper())

def get_raf_weight(hcc_id: int) -> float:
    """Return RAF weight for a given HCC category."""
    return HCC_CATEGORIES.get(hcc_id, ("Unknown", 0.0))[1]

def get_hcc_reference_df() -> pd.DataFrame:
    """Return HCC reference table as a DataFrame."""
    rows = []
    for hcc_id, (desc, raf) in HCC_CATEGORIES.items():
        rows.append({
            "hcc_id": hcc_id,
            "description": desc,
            "base_raf_weight": raf
        })
    return pd.DataFrame(rows)

def get_icd_hcc_mapping_df() -> pd.DataFrame:
    """Return ICD-10 to HCC mapping as a DataFrame."""
    rows = []
    for icd, hcc_id in ICD_TO_HCC.items():
        desc, raf = HCC_CATEGORIES.get(hcc_id, ("Unknown", 0.0))
        rows.append({
            "icd10_code": icd,
            "hcc_id": hcc_id,
            "hcc_description": desc,
            "raf_weight": raf
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=== HCC Reference Table ===")
    print(get_hcc_reference_df().to_string(index=False))
    print("\n=== ICD-10 → HCC Mapping ===")
    print(get_icd_hcc_mapping_df().to_string(index=False))
