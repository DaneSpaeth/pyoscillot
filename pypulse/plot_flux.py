import numpy as np
import matplotlib.pyplot as plt

fluxes = []
phases = []
with open("flux_test.txt", "r") as f:
    for line in f:
        columns = line.strip().split()
        try:
            phases.append(float(columns[0]))
            fluxes.append(float(columns[1]))
        except:
            pass


fluxes = np.array(fluxes)
phases = np.array(phases)
fluxes = fluxes / np.max(fluxes)

plt.plot(phases, fluxes, "bo")
plt.show()
