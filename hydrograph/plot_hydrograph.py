import matplotlib.pyplot as plt
import pandas as pd

hydro_df = pd.read_csv("hydrographs_output_2.csv")

basin = hydro_df[hydro_df["basin_id"] == 780002925]

plt.plot(basin["time_hr"], basin["flow_cms"])
plt.xlabel("Time (hr)")
plt.ylabel("Flow (cms)")
plt.title("Hydrograph")
plt.show()