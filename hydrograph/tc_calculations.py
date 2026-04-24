"""
Time of Concentration (Tc) Calculations
========================================
Computes 10 empirical Tc equations for each watershed in vpu_702_data.csv.

CSV Column Mapping:
  - L  = length_max   (meters  → converted to km where needed)
  - S  = slope        (m/m, dimensionless)
  - A  = area_km2     (km²)
  - H  = H            (meters, max - min elevation)
  - Hm = H            (same column used for mean relief in Giandotti)

All Tc results are output in HOURS.
"""

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = pd.read_csv("vpu_702_data.csv")

# ---------------------------------------------------------------------------
# Unit conversions
# length_max is in metres; most equations expect km
# ---------------------------------------------------------------------------
df["L_km"] = df["length_max"] / 1000.0   # metres → km
df["A"]    = df["area_km2"]               # already km²
df["S"]    = df["slope"]                  # dimensionless (m/m)
df["H"]    = df["H"]                      # metres
df["Hm"]   = df["elev_mean"] - df["elev_min"]        # metres (mean elev above outlet)

# Guard against zero / negative slope or H (would cause divide-by-zero / NaN)
S = df["S"].clip(lower=0.001)
L = df["L_km"]
A = df["A"]
H = df["H"].clip(lower=0.1)
Hm = df["Hm"].clip(lower=0.1)

# ---------------------------------------------------------------------------
# 1. Kirpich  –  Tc = 0.0663 * L^0.77 * S^-0.385      (hours)
# ---------------------------------------------------------------------------
df["Tc_Kirpich"] = 0.0663 * L**0.77 * S**(-0.385)

# ---------------------------------------------------------------------------
# 2. Ventura  –  Tc = 4 * A^0.5 * L^0.5 * H^-0.5      (hours)
# ---------------------------------------------------------------------------
df["Tc_Ventura"] = 4.0 * A**0.5 * L**0.5 * H**(-0.5)

# ---------------------------------------------------------------------------
# 3. Giandotti  –  Tc = (4*sqrt(A) + 3*L) / (0.8 * sqrt(H))   (hours)
# ---------------------------------------------------------------------------
df["Tc_Giandotti"] = (4.0 * np.sqrt(A) + 3.0 * L) / (0.8 * np.sqrt(Hm))

# ---------------------------------------------------------------------------
# 4. Pasini  –  Tc = 0.108 * A^0.333 * L^0.333 * S^-0.5       (hours)
# ---------------------------------------------------------------------------
df["Tc_Pasini"] = 0.108 * A**0.333 * L**0.333 * S**(-0.5)

# ---------------------------------------------------------------------------
# 5. Dooge  –  Tc = 0.365 * A^0.41 * S^-0.17                   (hours)
# ---------------------------------------------------------------------------
df["Tc_Dooge"] = 0.365 * A**0.41 * S**(-0.17)

# ---------------------------------------------------------------------------
# 6. Johnstone  –  Tc = 0.4623 * L^0.5 * S^-0.25               (hours)
# ---------------------------------------------------------------------------
df["Tc_Johnstone"] = 0.4623 * L**0.5 * S**(-0.25)

# ---------------------------------------------------------------------------
# 7. Corps of Engineers  –  Tc = 0.191 * L^0.76 * S^-0.19      (hours)
# ---------------------------------------------------------------------------
df["Tc_CorpsOfEngineers"] = 0.191 * L**0.76 * S**(-0.19)

# ---------------------------------------------------------------------------
# 8. Picking  –  Tc = 0.0883 * L^0.667 * S^-0.333              (hours)
# ---------------------------------------------------------------------------
df["Tc_Picking"] = 0.0883 * L**0.667 * S**(-0.333)

# ---------------------------------------------------------------------------
# 9. Temez  –  Tc = 0.3 * (L / S^0.25)^0.76                    (hours)
# ---------------------------------------------------------------------------
df["Tc_Temez"] = 0.3 * (L / S**0.25)**0.76

# ---------------------------------------------------------------------------
# 10. Bransby Williams  –  Tc = 0.605 * L / ((100*S)^0.2 * A^0.1)  (hours)
# ---------------------------------------------------------------------------
df["Tc_BransbyWilliams"] = 0.605 * L / ((100.0 * S)**0.2 * A**0.1)

# ---------------------------------------------------------------------------
# Assemble output
# ---------------------------------------------------------------------------
tc_cols = [
    "Tc_Kirpich", "Tc_Ventura", "Tc_Giandotti", "Tc_Pasini", "Tc_Dooge",
    "Tc_Johnstone", "Tc_CorpsOfEngineers", "Tc_Picking", "Tc_Temez",
    "Tc_BransbyWilliams",
]

id_cols = ["fid", "linkno"]
results = df[id_cols + tc_cols].copy()

# Optional: also compute a simple mean across all methods
results["Tc_Mean"] = results[tc_cols].mean(axis=1)

# ---------------------------------------------------------------------------
# Save to CSV
# ---------------------------------------------------------------------------
out_path = "tc_results.csv"
results.to_csv(out_path, index=False, float_format="%.4f")
print(f"Saved {len(results):,} rows → {out_path}")

# ---------------------------------------------------------------------------
# Quick sanity-check: print summary stats for each method
# ---------------------------------------------------------------------------
print("\n=== Tc Summary Statistics (hours) ===")
print(results[tc_cols + ["Tc_Mean"]].describe().round(3).to_string())