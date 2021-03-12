import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(0, 5, 1000)
ts = np.linspace(0, 1, 5)
nu = 1
k = 1
f = np.sin(2 * np.pi * k * x)

fig, ax = plt.subplots(len(ts))
for idx, t in enumerate(ts):
    f = np.sin(-2 * np.pi * k * x + 2 * np.pi * nu * t)
    ax[idx].plot(x, f)
plt.show()
