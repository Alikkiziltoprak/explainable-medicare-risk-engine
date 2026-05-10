"""
Revenue Leakage Detection Engine
----------------------------------
Identifies Medicare members with suspected missing HCC codes
that cause RAF score drops and revenue loss for the health plan.

Key outputs:
- Suspected missing HCC flags
- Estimated RAF delta per member
- Estimated PMPM and annual revenue leakage
- Provider coding accuracy scoring
- Leakage priority tiers (High / Medium / Low)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import os
import warnings

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_PATH  = "data/synthetic/medicare_members.csv"
OUTPUT_DIR = "outputs/reports"
PLOT_DIR   = "outputs/shap"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,   exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────
BASE_PMPM          = 850.0   # USD — simplified Medicare base rate
MONTHS_IN_YEAR     = 12
HIGH_LEAKAGE_PMPM  = 200.0   # threshold for High priority tier
MED_LEAKAGE_PMPM   = 75.0    # threshold for Medium priority tier

# HCC metadata for alert messages
HCC_META = {
    85:  {"name": "Congestive Heart Failure",      "icd_hint": "I50.xx"},
    18:  {"name": "Diabetes w/ Chronic Comp.",     "icd_hint": "E11.6x"},
    17:  {"name": "Diabetes w/ Acute Comp.",       "icd_hint": "E11.0x"},
    19:  {"name": "Diabetes w/o Complication",     "icd_hint": "E11.9"},
    111: {"name": "COPD",                          "icd_hint": "J44.x"},
    138: {"name": "CKD Stage 3 (Moderate)",        "icd_hint": "N18.3"},
    137: {"name": "CKD Stage 4 (Severe)",          "icd_hint": "N18.4"},
    136: {"name": "CKD Stage 5",                   "icd_hint": "N18.5"},
    96:  {"name": "Specified Heart Arrhythmias",   "icd_hint": "I48.xx"},
    108: {"name": "Vascular Disease w/ Comp.",     "icd_hint": "I70.xx"},
    57:  {"name": "Schizophrenia",                 "icd_hint": "F20.x"},
    58:  {"name": "Major Depression/Bipolar",      "icd_hint": "F31-F33"},
}

# RAF weights for leakage calculation
RAF_WEIGHTS = {
    85: 0.331, 18: 0.318, 17: 0.318, 19: 0.105,
    111: 0.328, 138: 0.069, 137: 0.289, 136: 0.289,
    96: 0.179, 108: 0.539, 57: 0.421, 58: 0.212,
}


# ── 1. Load Data ───────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


# ── 2. Detect Missing HCCs ────────────────────────────────────────────────
def detect_missing_hccs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare prior year HCCs vs current year.
    Flag members where chronic conditions present in prior year
    are missing in current year — indicating potential coding gaps.
    """
    results = []

    for _, row in df.iterrows():
        prior_hccs = set()
        current_hccs = set()

        if pd.notna(row["prior_hcc_list"]) and str(row["prior_hcc_list"]) != "":
            try:
                prior_hccs = {int(x) for x in str(row["prior_hcc_list"]).split("|") if x.strip()}
            except:
                pass

        if pd.notna(row["current_hcc_list"]) and str(row["current_hcc_list"]) != "":
            try:
                current_hccs = {int(x) for x in str(row["current_hcc_list"]).split("|") if x.strip()}
            except:
                pass

        missing_hccs = prior_hccs - current_hccs

        # Only flag chronic/persistent conditions (not acute)
        chronic_hccs = {85, 18, 17, 19, 111, 138, 137, 136, 96, 108, 57, 58}
        flagged_hccs = missing_hccs & chronic_hccs

        total_raf_delta  = sum(RAF_WEIGHTS.get(h, 0.0) for h in flagged_hccs)
        estimated_pmpm   = round(total_raf_delta * BASE_PMPM, 2)
        estimated_annual = round(estimated_pmpm * MONTHS_IN_YEAR, 2)

        # Priority tier
        if estimated_pmpm >= HIGH_LEAKAGE_PMPM:
            priority = "High"
        elif estimated_pmpm >= MED_LEAKAGE_PMPM:
            priority = "Medium"
        elif estimated_pmpm > 0:
            priority = "Low"
        else:
            priority = "None"

        # Alert messages
        alerts = []
        for hcc in sorted(flagged_hccs):
            meta = HCC_META.get(hcc, {"name": f"HCC {hcc}", "icd_hint": "N/A"})
            alerts.append(
                f"HCC {hcc} ({meta['name']}) present in prior year — "
                f"suggest coding {meta['icd_hint']} | "
                f"RAF impact: +{RAF_WEIGHTS.get(hcc, 0):.3f} | "
                f"Est. PMPM: +${RAF_WEIGHTS.get(hcc, 0)*BASE_PMPM:.2f}"
            )

        results.append({
            "member_id":          row["member_id"],
            "age":                row["age"],
            "gender":             row["gender"],
            "dual_eligible":      row["dual_eligible"],
            "region":             row["region"],
            "disease_profile":    row["disease_profile"],
            "prior_raf":          row["prior_raf_score"],
            "current_raf":        row["current_raf_score"],
            "raf_delta":          round(row["prior_raf_score"] - row["current_raf_score"], 4),
            "missing_hcc_count":  len(flagged_hccs),
            "missing_hccs":       "|".join(map(str, sorted(flagged_hccs))),
            "estimated_pmpm_loss":estimated_pmpm,
            "estimated_annual_loss": estimated_annual,
            "priority_tier":      priority,
            "coding_alerts":      " || ".join(alerts) if alerts else "No gaps detected",
            "action_required":    len(flagged_hccs) > 0,
        })

    return pd.DataFrame(results)


# ── 3. Provider Coding Accuracy ────────────────────────────────────────────
def provider_coding_accuracy(leakage_df: pd.DataFrame) -> pd.DataFrame:
    """
    Score each region's coding accuracy based on leakage rate and RAF delta.
    In a real scenario this would be by provider/group — here we use region as proxy.
    """
    region_stats = leakage_df.groupby("region").agg(
        total_members      = ("member_id", "count"),
        members_with_gap   = ("action_required", "sum"),
        total_pmpm_leakage = ("estimated_pmpm_loss", "sum"),
        avg_raf_delta      = ("raf_delta", "mean"),
    ).reset_index()

    region_stats["gap_rate_pct"] = round(
        region_stats["members_with_gap"] / region_stats["total_members"] * 100, 1)
    region_stats["coding_accuracy_score"] = round(
        100 - region_stats["gap_rate_pct"], 1)
    region_stats["annual_leakage_est"] = round(
        region_stats["total_pmpm_leakage"] * MONTHS_IN_YEAR, 2)

    return region_stats.sort_values("coding_accuracy_score")


# ── 4. Leakage Summary ────────────────────────────────────────────────────
def leakage_summary(leakage_df: pd.DataFrame) -> dict:
    flagged = leakage_df[leakage_df["action_required"]]
    return {
        "total_members":          len(leakage_df),
        "members_flagged":        len(flagged),
        "flag_rate_pct":          round(len(flagged) / len(leakage_df) * 100, 1),
        "total_monthly_leakage":  round(flagged["estimated_pmpm_loss"].sum(), 2),
        "total_annual_leakage":   round(flagged["estimated_annual_loss"].sum(), 2),
        "avg_pmpm_per_flagged":   round(flagged["estimated_pmpm_loss"].mean(), 2),
        "high_priority_count":    len(flagged[flagged["priority_tier"] == "High"]),
        "medium_priority_count":  len(flagged[flagged["priority_tier"] == "Medium"]),
        "low_priority_count":     len(flagged[flagged["priority_tier"] == "Low"]),
    }


# ── 5. Visualizations ─────────────────────────────────────────────────────
def plot_leakage_dashboard(leakage_df: pd.DataFrame,
                           provider_df: pd.DataFrame,
                           summary: dict,
                           save_path: str):
    flagged = leakage_df[leakage_df["action_required"]]

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Medicare Revenue Leakage Detection Dashboard",
                 fontsize=16, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    COLORS = {
        "High":   "#C44E52",
        "Medium": "#DD8452",
        "Low":    "#4C72B0",
        "None":   "#CCCCCC"
    }

    # ── Panel 1: Priority Tier Distribution ──
    ax1 = fig.add_subplot(gs[0, 0])
    tier_counts = leakage_df["priority_tier"].value_counts()
    tier_order  = ["High", "Medium", "Low", "None"]
    tier_vals   = [tier_counts.get(t, 0) for t in tier_order]
    tier_colors = [COLORS[t] for t in tier_order]
    wedges, texts, autotexts = ax1.pie(
        tier_vals, labels=tier_order, colors=tier_colors,
        autopct="%1.1f%%", startangle=140,
        textprops={"fontsize": 9})
    ax1.set_title("Members by Leakage Priority Tier", fontweight="bold", fontsize=11)

    # ── Panel 2: Monthly PMPM Leakage by Region ──
    ax2 = fig.add_subplot(gs[0, 1])
    region_sorted = provider_df.sort_values("total_pmpm_leakage", ascending=True)
    bars = ax2.barh(region_sorted["region"], region_sorted["total_pmpm_leakage"],
                    color="#C44E52", edgecolor="white")
    ax2.set_xlabel("Total Monthly PMPM Leakage ($)", fontsize=9)
    ax2.set_title("PMPM Revenue Leakage by Region", fontweight="bold", fontsize=11)
    for bar, val in zip(bars, region_sorted["total_pmpm_leakage"]):
        ax2.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                 f"${val:,.0f}", va="center", fontsize=8)

    # ── Panel 3: Coding Accuracy Score by Region ──
    ax3 = fig.add_subplot(gs[0, 2])
    acc_sorted = provider_df.sort_values("coding_accuracy_score")
    bar_colors = ["#C44E52" if s < 95 else "#55A868"
                  for s in acc_sorted["coding_accuracy_score"]]
    bars3 = ax3.barh(acc_sorted["region"], acc_sorted["coding_accuracy_score"],
                     color=bar_colors, edgecolor="white")
    ax3.set_xlim(85, 102)
    ax3.axvline(x=95, color="orange", linestyle="--", linewidth=1.5,
                label="95% Threshold")
    ax3.set_xlabel("Coding Accuracy Score (%)", fontsize=9)
    ax3.set_title("Provider Coding Accuracy by Region", fontweight="bold", fontsize=11)
    ax3.legend(fontsize=8)
    for bar, val in zip(bars3, acc_sorted["coding_accuracy_score"]):
        ax3.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}%", va="center", fontsize=8)

    # ── Panel 4: RAF Delta Distribution (flagged members) ──
    ax4 = fig.add_subplot(gs[1, 0])
    if len(flagged) > 0:
        ax4.hist(flagged["raf_delta"], bins=20, color="#DD8452",
                 edgecolor="white", linewidth=0.8)
        ax4.axvline(flagged["raf_delta"].mean(), color="#C44E52",
                    linestyle="--", linewidth=2,
                    label=f"Mean: {flagged['raf_delta'].mean():.3f}")
        ax4.set_xlabel("RAF Delta (Prior − Current)", fontsize=9)
        ax4.set_ylabel("Member Count", fontsize=9)
        ax4.set_title("RAF Score Drop Distribution\n(Flagged Members Only)",
                      fontweight="bold", fontsize=11)
        ax4.legend(fontsize=8)

    # ── Panel 5: Top Missing HCCs ──
    ax5 = fig.add_subplot(gs[1, 1])
    hcc_counts = {}
    for hccs in flagged["missing_hccs"]:
        if pd.notna(hccs) and hccs:
            for h in str(hccs).split("|"):
                if h.strip():
                    hcc_id = int(h.strip())
                    name = HCC_META.get(hcc_id, {}).get("name", f"HCC {hcc_id}")
                    label = f"HCC {hcc_id}\n{name[:20]}"
                    hcc_counts[label] = hcc_counts.get(label, 0) + 1

    if hcc_counts:
        hcc_df = pd.DataFrame(list(hcc_counts.items()),
                               columns=["HCC", "Count"]).sort_values("Count")
        ax5.barh(hcc_df["HCC"], hcc_df["Count"],
                 color="#4C72B0", edgecolor="white")
        ax5.set_xlabel("Members Affected", fontsize=9)
        ax5.set_title("Top Missing HCC Categories", fontweight="bold", fontsize=11)
        ax5.tick_params(axis="y", labelsize=7)

    # ── Panel 6: KPI Summary Box ──
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")

    kpi_lines = [
        ("Total Members Analyzed",    f"{summary['total_members']:,}"),
        ("Members Flagged",           f"{summary['members_flagged']:,} ({summary['flag_rate_pct']}%)"),
        ("High Priority",             f"{summary['high_priority_count']:,} members"),
        ("Medium Priority",           f"{summary['medium_priority_count']:,} members"),
        ("Monthly PMPM Leakage",      f"${summary['total_monthly_leakage']:,.2f}"),
        ("Estimated Annual Leakage",  f"${summary['total_annual_leakage']:,.2f}"),
        ("Avg Loss per Flagged Mbr",  f"${summary['avg_pmpm_per_flagged']:,.2f}/mo"),
    ]

    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.add_patch(mpatches.FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96,
        boxstyle="round,pad=0.02",
        facecolor="#F8F9FA", edgecolor="#CCCCCC", linewidth=1.5))

    ax6.text(0.5, 0.93, "📊 Leakage Summary", ha="center", va="top",
             fontsize=12, fontweight="bold", color="#2C3E50")
    ax6.axhline(y=0.88, xmin=0.05, xmax=0.95, color="#CCCCCC", linewidth=0.8)

    for i, (label, value) in enumerate(kpi_lines):
        y_pos = 0.80 - i * 0.11
        ax6.text(0.08, y_pos, label + ":", fontsize=9, color="#555555", va="center")
        color = "#C44E52" if "Leakage" in label or "Annual" in label else "#2C3E50"
        ax6.text(0.92, y_pos, value, fontsize=9, fontweight="bold",
                 color=color, va="center", ha="right")

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"  ✅ Leakage dashboard saved → {save_path}")


# ── MAIN ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  MEDICARE REVENUE LEAKAGE DETECTION ENGINE")
    print("="*55)

    print("\n[1/4] Loading member data...")
    df = load_data(DATA_PATH)
    print(f"  Loaded {len(df):,} members")

    print("\n[2/4] Detecting missing HCC codes...")
    leakage_df = detect_missing_hccs(df)
    flagged    = leakage_df[leakage_df["action_required"]]
    print(f"  Members flagged: {len(flagged):,}")

    print("\n[3/4] Scoring provider coding accuracy...")
    provider_df = provider_coding_accuracy(leakage_df)

    print("\n[4/4] Generating leakage dashboard...")
    summary = leakage_summary(leakage_df)
    plot_leakage_dashboard(leakage_df, provider_df,
                           summary,
                           f"{PLOT_DIR}/leakage_dashboard.png")

    # Save reports
    leakage_df.to_csv(f"{OUTPUT_DIR}/leakage_report.csv", index=False)
    provider_df.to_csv(f"{OUTPUT_DIR}/provider_coding_accuracy.csv", index=False)
    print(f"  ✅ Leakage report saved → {OUTPUT_DIR}/leakage_report.csv")
    print(f"  ✅ Provider accuracy report saved → {OUTPUT_DIR}/provider_coding_accuracy.csv")

    # Print summary
    print("\n" + "="*55)
    print("  LEAKAGE SUMMARY")
    print("="*55)
    for k, v in summary.items():
        label = k.replace("_", " ").title()
        if "leakage" in k or "annual" in k:
            print(f"  {label:<30} ${v:>12,.2f}" if isinstance(v, float) else
                  f"  {label:<30} {v:>12}")
        else:
            print(f"  {label:<30} {str(v):>12}")

    print("\n  Provider Coding Accuracy:")
    print(provider_df[["region", "total_members", "members_with_gap",
                        "gap_rate_pct", "coding_accuracy_score",
                        "annual_leakage_est"]].to_string(index=False))

    print("\n  Sample High-Priority Alerts:")
    high = leakage_df[leakage_df["priority_tier"] == "High"].head(3)
    for _, row in high.iterrows():
        print(f"\n  Member: {row['member_id']} | Age: {row['age']} | "
              f"Region: {row['region']}")
        print(f"  RAF: {row['prior_raf']:.3f} → {row['current_raf']:.3f} "
              f"(Δ {row['raf_delta']:.3f})")
        print(f"  Est. PMPM Loss: ${row['estimated_pmpm_loss']:,.2f} | "
              f"Annual: ${row['estimated_annual_loss']:,.2f}")
        print(f"  Alert: {row['coding_alerts'][:120]}...")
