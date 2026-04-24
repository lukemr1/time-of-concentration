import pandas as pd
import matplotlib.pyplot as plt

vpu_702_df = pd.read_csv("vpu_702_data.csv")
script_df = pd.read_csv("tc_zonal_stats_test.csv")


merged = vpu_702_df.merge(script_df, on="linkno")
merged = merged.dropna(subset=["length_max"])

merged["dist_diff"] = merged["length_max"] - merged["max_dist_to_stream"]
merged["dist_pct_diff"] = merged["dist_diff"] / merged["length_max"] * 100

merged["elev_diff"] = merged["elev_max"] - merged["max_elev"]

top50 = merged.sort_values("dist_pct_diff", key=abs, ascending=False).head(50)

plt.scatter(
    merged["length_max"],
    merged["max_dist_to_stream"]
)
plt.xlabel("Full VPU")
plt.ylabel("Tile")
plt.title("Distance Comparison")
plt.show()

print(merged)
print(top50)