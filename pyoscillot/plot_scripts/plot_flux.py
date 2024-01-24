import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

root = Path("/home/dane/mounted_srv/simulations/fake_spectra")

simulations = ["EV_Lac_2SPOTS_SAME_SIZE_EQUATOR_OPPOSITE",
               "EV_Lac_2SPOTS_SAME_SIZE_EQUATOR_CLOSER",
               "EV_Lac_2SPOTS_DIFF_SIZE_EQUATOR_OPPOSITE",
               "EV_Lac_2SPOTS_SAME_SIZE_ONE45_OPPOSITE",
               "EV_Lac_EQUATOR+POLE"]

colors = ["blue", "red", "green", "purple", "orange"]

simulations = ["EV_Lac_15Spots"]
colors = ["blue"]

P_rot = 4.3491
fig, ax = plt.subplots(1, figsize=(16,9))
for sim, color in zip(simulations, colors):
    fluxes = []
    phases = []
    with open(root / sim / "flux.txt", "r") as f:
        for line in f:
            columns = line.strip().split()
            try:
                phases.append(float(columns[0]))
                fluxes.append(float(columns[1]))
            except:
                pass
    fluxes = np.array(fluxes)
    phases = np.array(phases)
    phases = np.mod(phases, P_rot)
    fluxes = fluxes / np.max(fluxes)

    ax.scatter(phases, fluxes, color=color, label=sim.replace("EV_LAC", ""))

ax.set_xlabel("Rotational Phase [d]")
ax.set_ylabel("Normalized Flux")
ax.set_xlim(0, P_rot)

ax.legend()
fig.set_tight_layout(True)
plt.savefig("/home/dane/Documents/PhD/pyoscillot/plots/EV_Lac_15_spots_flux.png")
