"""
Synthetic Medicare Member Data Generator
-----------------------------------------
Generates realistic Medicare Advantage member population with:
- Demographics (age, gender, dual eligibility, region)
- Diagnosis history (ICD-10 codes mapped to HCCs)
- RAF score calculation
- Intentional missing HCC scenarios for leakage detection
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.hcc_reference import ICD_TO_HCC, HCC_CATEGORIES, get_raf_weight

# ── Seed for reproducibility ───────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ── Constants ──────────────────────────────────────────────────────────────
N_MEMBERS = 2000

DEMOGRAPHIC_RAF_FACTORS = {
    # Age band → additive RAF factor (simplified CMS age/sex table)
    (65, 69): {"M": 0.346, "F": 0.322},
    (70, 74): {"M": 0.422, "F": 0.391},
    (75, 79): {"M": 0.535, "F": 0.492},
    (80, 84): {"M": 0.643, "F": 0.601},
    (85, 99): {"M": 0.751, "F": 0.712},
}

DUAL_ELIGIBILITY_BONUS = 0.142  # CMS dual status RAF uplift

DISEASE_PROFILES = [
    # (profile_name, icd_list, prevalence_weight)
    ("Diabetic+CKD",        ["E1165", "E1152", "N184"],           0.18),
    ("CHF+Arrhythmia",      ["I5032", "I4891"],                   0.12),
    ("Diabetic+CHF+CKD",    ["E1165", "I5032", "N184"],           0.10),
    ("COPD",                ["J449"],                              0.14),
    ("Cancer_Lung",         ["C3490"],                             0.04),
    ("Cancer_Breast",       ["C509"],                              0.05),
    ("CKD_Moderate",        ["N183"],                              0.09),
    ("Schizophrenia",       ["F209"],                              0.03),
    ("Depression",          ["F329"],                              0.07),
    ("Vascular",            ["I7000"],                             0.06),
    ("Healthy",             [],                                    0.12),
]

REGIONS = ["Northeast", "Southeast", "Midwest", "Southwest", "West"]

# ── Leakage Scenarios ──────────────────────────────────────────────────────
# Members who HAD a condition in prior year but it's missing in current year
LEAKAGE_RATE = 0.20  # 20% of members will have at least one missing HCC

LEAKAGE_SCENARIOS = [
    {
        "name": "CKD_Not_Coded",
        "prior_icd": "N183",
        "missing_hcc": 138,
        "description": "CKD Stage 3 present in prior year, not coded this year"
    },
    {
        "name": "CHF_Not_Coded",
        "prior_icd": "I5032",
        "missing_hcc": 85,
        "description": "CHF present in prior year, not coded this year"
    },
    {
        "name": "Diabetes_Downgraded",
        "prior_icd": "E1165",
        "missing_hcc": 18,
        "description": "Diabetic with complications downgraded to no-complication"
    },
    {
        "name": "COPD_Not_Coded",
        "prior_icd": "J449",
        "missing_hcc": 111,
        "description": "COPD present in prior year, not coded this year"
    },
]


def get_age_sex_raf(age: int, gender: str) -> float:
    for (low, high), factors in DEMOGRAPHIC_RAF_FACTORS.items():
        if low <= age <= high:
            return factors.get(gender, 0.35)
    return 0.35


def assign_disease_profile() -> tuple:
    profiles = [p[0] for p in DISEASE_PROFILES]
    weights = [p[2] for p in DISEASE_PROFILES]
    weights = np.array(weights) / sum(weights)
    chosen_idx = np.random.choice(len(DISEASE_PROFILES), p=weights)
    return DISEASE_PROFILES[chosen_idx]


def calculate_raf_score(age: int, gender: str, dual: bool, hcc_list: list) -> float:
    score = get_age_sex_raf(age, gender)
    if dual:
        score += DUAL_ELIGIBILITY_BONUS
    for hcc_id in set(hcc_list):
        score += get_raf_weight(hcc_id)
    # Interaction factors (simplified)
    if 85 in hcc_list and 18 in hcc_list:   # CHF + Diabetes
        score += 0.121
    if 85 in hcc_list and 138 in hcc_list:  # CHF + CKD
        score += 0.156
    if 18 in hcc_list and 138 in hcc_list:  # Diabetes + CKD
        score += 0.099
    return round(score, 4)


def generate_member_data(n: int = N_MEMBERS) -> pd.DataFrame:
    records = []

    for i in range(n):
        member_id = f"MBR{100000 + i}"
        age = int(np.random.choice(
            range(65, 95),
            p=np.array([max(0.001, 1 / (1 + abs(a - 74))) for a in range(65, 95)]) /
              sum([max(0.001, 1 / (1 + abs(a - 74))) for a in range(65, 95)])
        ))
        gender = np.random.choice(["M", "F"], p=[0.45, 0.55])
        dual_eligible = np.random.choice([True, False], p=[0.22, 0.78])
        region = np.random.choice(REGIONS)

        # Assign disease profile
        profile_name, icd_codes, _ = assign_disease_profile()

        # Map ICD → HCC
        hcc_list = []
        for icd in icd_codes:
            hcc = ICD_TO_HCC.get(icd)
            if hcc:
                hcc_list.append(hcc)

        # Prior year HCCs (used for leakage detection)
        prior_hcc_list = hcc_list.copy()
        prior_icd_codes = icd_codes.copy()

        # Introduce leakage
        has_leakage = False
        leakage_scenario = None
        missing_hcc = None
        missing_raf_delta = 0.0

        if np.random.random() < LEAKAGE_RATE and hcc_list:
            scenario = random.choice(LEAKAGE_SCENARIOS)
            prior_hcc = scenario["missing_hcc"]
            if prior_hcc in hcc_list:
                hcc_list.remove(prior_hcc)
                has_leakage = True
                leakage_scenario = scenario["name"]
                missing_hcc = prior_hcc
                missing_raf_delta = get_raf_weight(prior_hcc)

        # RAF scores
        current_raf = calculate_raf_score(age, gender, dual_eligible, hcc_list)
        prior_raf = calculate_raf_score(age, gender, dual_eligible, prior_hcc_list)

        # PMPM estimate (simplified: RAF × base rate)
        BASE_PMPM = 850  # USD, simplified Medicare base rate
        current_pmpm = round(current_raf * BASE_PMPM, 2)
        prior_pmpm = round(prior_raf * BASE_PMPM, 2)
        pmpm_leakage = round(prior_pmpm - current_pmpm, 2) if has_leakage else 0.0

        records.append({
            "member_id":          member_id,
            "age":                age,
            "gender":             gender,
            "dual_eligible":      int(dual_eligible),
            "region":             region,
            "disease_profile":    profile_name,
            "current_icd_codes":  "|".join(icd_codes) if not has_leakage else
                                  "|".join([c for c in icd_codes
                                            if ICD_TO_HCC.get(c) != missing_hcc]),
            "prior_icd_codes":    "|".join(prior_icd_codes),
            "current_hcc_list":   "|".join(map(str, sorted(set(hcc_list)))),
            "prior_hcc_list":     "|".join(map(str, sorted(set(prior_hcc_list)))),
            "hcc_count":          len(set(hcc_list)),
            "current_raf_score":  current_raf,
            "prior_raf_score":    prior_raf,
            "raf_delta":          round(prior_raf - current_raf, 4),
            "current_pmpm":       current_pmpm,
            "prior_pmpm":         prior_pmpm,
            "pmpm_leakage":       pmpm_leakage,
            "has_leakage":        int(has_leakage),
            "leakage_scenario":   leakage_scenario if has_leakage else "None",
            "missing_hcc":        missing_hcc if has_leakage else None,
            "missing_raf_delta":  missing_raf_delta,
        })

    return pd.DataFrame(records)


def save_data(df: pd.DataFrame, output_dir: str = "data/synthetic"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "medicare_members.csv")
    df.to_csv(path, index=False)
    print(f"✅ Saved {len(df)} member records → {path}")
    return path


def print_summary(df: pd.DataFrame):
    print("\n" + "="*55)
    print("  SYNTHETIC MEDICARE MEMBER DATA — SUMMARY")
    print("="*55)
    print(f"  Total Members       : {len(df):,}")
    print(f"  Avg Age             : {df['age'].mean():.1f}")
    print(f"  Dual Eligible       : {df['dual_eligible'].mean()*100:.1f}%")
    print(f"  Avg RAF Score       : {df['current_raf_score'].mean():.3f}")
    print(f"  Avg PMPM            : ${df['current_pmpm'].mean():,.2f}")
    print(f"  Members w/ Leakage  : {df['has_leakage'].sum():,} ({df['has_leakage'].mean()*100:.1f}%)")
    print(f"  Total PMPM Leakage  : ${df['pmpm_leakage'].sum():,.2f}/month")
    print(f"  Annual Leakage Est. : ${df['pmpm_leakage'].sum()*12:,.2f}")
    print("="*55)

    print("\n  Disease Profile Distribution:")
    profile_counts = df['disease_profile'].value_counts()
    for profile, count in profile_counts.items():
        print(f"    {profile:<25} {count:>5} ({count/len(df)*100:.1f}%)")

    print("\n  Leakage Scenarios:")
    leakage_df = df[df['has_leakage'] == 1]
    if len(leakage_df) > 0:
        scenario_counts = leakage_df['leakage_scenario'].value_counts()
        for scenario, count in scenario_counts.items():
            avg_loss = leakage_df[leakage_df['leakage_scenario'] == scenario]['pmpm_leakage'].mean()
            print(f"    {scenario:<25} {count:>4} members | Avg PMPM Loss: ${avg_loss:.2f}")


if __name__ == "__main__":
    print("Generating synthetic Medicare member data...")
    df = generate_member_data(N_MEMBERS)
    print_summary(df)
    save_data(df)
    print("\nSample records:")
    print(df[["member_id","age","gender","dual_eligible","disease_profile",
              "current_raf_score","has_leakage","pmpm_leakage"]].head(10).to_string(index=False))
