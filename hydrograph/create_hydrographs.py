"""
SCS Dimensionless Unit Hydrograph Comparison
=============================================
For each of 3 representative watersheds (small, medium, large), this script:
  1. Reads Tc values computed by all 10 methods from tc_results.csv
  2. Flags any Tc > 36 hours as unreliable
  3. Builds an SCS unit hydrograph for each valid Tc
  4. Plots all 10 UHs on the same axes (one figure per watershed)

SCS Method
----------
The SCS dimensionless unit hydrograph is defined by the ratio table (t/Tp vs q/qp).
Given Tc, the time to peak is:
    Tp = 0.6 * Tc  (SCS approximation for the rising limb lag)

Peak discharge per unit area (m³/s per km² per mm of runoff):
    qp = 0.208 * A / Tp        (A in km², Tp in hours → qp in m³/s/mm)

The dimensionless SCS UH coordinates (t/Tp, q/qp) are fixed tabulated values.
Actual hydrograph: t = (t/Tp)*Tp,  q = (q/qp)*qp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# SCS dimensionless UH table  (t/Tp, q/qp)
# ---------------------------------------------------------------------------
SCS_T_RATIO = np.array([
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
    2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.5, 4.0, 4.5, 5.0
])
SCS_Q_RATIO = np.array([
    0.000, 0.030, 0.100, 0.190, 0.310, 0.470, 0.660, 0.820, 0.930, 0.990,
    1.000, 0.990, 0.930, 0.860, 0.780, 0.680, 0.560, 0.460, 0.390, 0.330,
    0.280, 0.207, 0.147, 0.107, 0.077, 0.055, 0.025, 0.011, 0.005, 0.000
])

TC_FLAG_THRESHOLD = 36.0   # hours — flag values above this

METHODS = [
    "Tc_Kirpich", "Tc_Ventura", "Tc_Giandotti", "Tc_Pasini", "Tc_Dooge",
    "Tc_Johnstone", "Tc_CorpsOfEngineers", "Tc_Picking", "Tc_Temez",
    "Tc_BransbyWilliams",
]
METHOD_LABELS = [
    "Kirpich", "Ventura", "Giandotti", "Pasini", "Dooge",
    "Johnstone", "Corps of Engineers", "Picking", "Temez", "Bransby Williams",
]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
tc_df   = pd.read_csv("tc_results.csv")
orig_df = pd.read_csv("vpu_702_data.csv")

# ---------------------------------------------------------------------------
# Select representative watersheds: Randomly sample within size tiers
# ---------------------------------------------------------------------------
orig_sorted = orig_df.sort_values("area_km2").reset_index(drop=True)
n = len(orig_sorted)

# Define pools (bottom 20%, middle 20%, top 20%) and sample 1 random row from each
rep_rows = {
    "Small":  orig_sorted.iloc[:int(0.2 * n)].sample(n=1).iloc[0],
    "Medium": orig_sorted.iloc[int(0.4 * n):int(0.6 * n)].sample(n=1).iloc[0],
    "Large":  orig_sorted.iloc[int(0.8 * n):].sample(n=1).iloc[0],
}

# ---------------------------------------------------------------------------
# Color palette — one color per method, consistent across all plots
# ---------------------------------------------------------------------------
colors = cm.tab10(np.linspace(0, 1, len(METHODS)))

# ---------------------------------------------------------------------------
# Helper: build SCS UH given Tc (hours) and area (km²)
# Returns (time_array_hours, discharge_array_m3_s_per_mm)
# ---------------------------------------------------------------------------
def scs_uh(Tc_hr, area_km2):
    Tp = 0.6 * Tc_hr                          # time to peak (hours)
    qp = 0.208 * area_km2 / Tp                # peak unit discharge (m³/s/mm)
    t  = SCS_T_RATIO * Tp                     # actual time (hours)
    q  = SCS_Q_RATIO * qp                     # actual discharge
    return t, q

# ---------------------------------------------------------------------------
# Plot — one figure per watershed
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
fig.suptitle(
    "SCS Unit Hydrographs — 10 Tc Methods\n(dashed = Tc > 36 h, flagged as unreliable)",
    fontsize=13, fontweight="bold", y=1.01
)

for ax, (ws_label, orig_row) in zip(axes, rep_rows.items()):
    fid      = int(orig_row["fid"])
    area_km2 = orig_row["area_km2"]
    L_km     = orig_row["length_max"] / 1000.0
    slope    = orig_row["slope"]
    H_m      = orig_row["H"]
    Hm_m     = orig_row["elev_mean"] - orig_row["elev_min"]
    elev_min = orig_row["elev_min"]
    elev_max = orig_row["elev_max"]
    tc_row   = tc_df[tc_df["fid"] == fid].iloc[0]

    flagged_methods = []

    for col, label, color in zip(METHODS, METHOD_LABELS, colors):
        Tc = tc_row[col]

        is_flagged = (Tc > TC_FLAG_THRESHOLD) or pd.isna(Tc)

        if is_flagged:
            flagged_methods.append(f"{label} (Tc={Tc:.1f}h)")
            linestyle = "--"
            alpha     = 0.45
            lw        = 1.2
        else:
            linestyle = "-"
            alpha     = 0.85
            lw        = 1.8

        t, q = scs_uh(Tc, area_km2)
        ax.plot(t, q, color=color, linestyle=linestyle,
                linewidth=lw, alpha=alpha, label=label)

    ax.set_title(
        f"{ws_label} Basin  (fid {fid})",
        fontsize=10, fontweight="bold"
    )
    ax.set_xlabel("Time (hours)", fontsize=9)
    ax.set_ylabel("Discharge (m³/s per mm of runoff)", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.tick_params(labelsize=8)

    # --- Input parameters info box (upper left) ---
    info_lines = [
        f"Area:       {area_km2:.2f} km²",
        f"Length (L): {L_km:.2f} km",
        f"Slope (S):  {slope:.4f} m/m",
        f"H (relief): {H_m:.1f} m",
        f"Hm (mean):  {Hm_m:.1f} m",
        f"Elev range: {elev_min:.0f}–{elev_max:.0f} m",
    ]
    ax.text(0.97, 0.97, "\n".join(info_lines),
            transform=ax.transAxes, fontsize=7,
            verticalalignment="top", horizontalalignment="right",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#eef4fb",
                      edgecolor="#4a90d9", alpha=0.92))

    if flagged_methods:
        flag_text = "⚑ Flagged (Tc > 36h):\n" + "\n".join(flagged_methods)
        ax.text(0.03, 0.97, flag_text,
                transform=ax.transAxes, fontsize=6.5,
                verticalalignment="top", horizontalalignment="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3cd",
                          edgecolor="#e0a800", alpha=0.9))

# Shared legend below all subplots
legend_elements = [
    Line2D([0], [0], color=colors[i], linewidth=2, label=METHOD_LABELS[i])
    for i in range(len(METHODS))
]
legend_elements += [
    Line2D([0], [0], color="gray", linewidth=1.5, linestyle="--",
           label="Flagged (Tc > 36h)"),
]
fig.legend(
    handles=legend_elements,
    loc="lower center", ncol=6,
    fontsize=8, frameon=True,
    bbox_to_anchor=(0.5, -0.08)
)

plt.tight_layout()
plt.savefig(
    "scs_unit_hydrographs_3.png",
    dpi=150, bbox_inches="tight"
)
print("Saved → scs_unit_hydrographs.png")

# ---------------------------------------------------------------------------
# Print summary of flagged Tc values across ALL watersheds
# ---------------------------------------------------------------------------
print(f"\n=== Flagged Tc values (> {TC_FLAG_THRESHOLD}h) across all {len(tc_df):,} watersheds ===")
for col, label in zip(METHODS, METHOD_LABELS):
    n_flagged = (tc_df[col] > TC_FLAG_THRESHOLD).sum()
    pct = 100 * n_flagged / len(tc_df)
    print(f"  {label:<22}: {n_flagged:>5,} flagged  ({pct:.1f}%)")