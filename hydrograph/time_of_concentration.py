import pandas as pd
import numpy as np

# -----------------------------
# USER INPUT
# -----------------------------
input_csv = "cleaned_catchments_2.csv"  # export from QGIS
output_csv = "hydrographs_output_2.csv"

D = 60  # rainfall duration in minutes

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(input_csv)

# -----------------------------
# COMPUTE HYDROGRAPH PARAMETERS
# -----------------------------
df["Tlag_min"] = 0.6 * df["Tc_final"]
df["Tp_min"] = df["Tlag_min"] + (D / 2)
df["Tp_hr"] = df["Tp_min"] / 60

df["Qp"] = (0.208 * df["area"]) / df["Tp_hr"]

# -----------------------------
# SCS DIMENSIONLESS CURVE
# -----------------------------
# Time ratios (t/Tp)
t_ratio = np.array([
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9, 1.0,
    1.2, 1.4, 1.6, 1.8, 2.0,
    2.2, 2.4, 2.67
])

# Flow ratios (Q/Qp)
q_ratio = np.array([
    0.0, 0.03, 0.1, 0.19, 0.3, 0.47,
    0.66, 0.82, 0.93, 0.99, 1.0,
    0.93, 0.78, 0.62, 0.48, 0.36,
    0.26, 0.18, 0.0
])

# -----------------------------
# GENERATE HYDROGRAPHS
# -----------------------------
hydrographs = []

for idx, row in df.iterrows():
    basin_id = row.get("linkno", idx)
    Tp = row["Tp_hr"]
    Qp = row["Qp"]

    # actual time (hours)
    time = t_ratio * Tp

    # actual discharge
    flow = q_ratio * Qp

    for t, q in zip(time, flow):
        hydrographs.append({
            "basin_id": basin_id,
            "time_hr": t,
            "flow_cms": q
        })

# -----------------------------
# SAVE OUTPUT
# -----------------------------
hydro_df = pd.DataFrame(hydrographs)
hydro_df.to_csv(output_csv, index=False)

print("Hydrographs generated for all basins!")