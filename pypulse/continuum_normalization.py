import numpy as np
import matplotlib.pyplot as plt
from dataloader import phoenix_spectrum
from physics import planck



T = 5800
wave, spec, header = phoenix_spectrum(T, 4.5, 0.0, wavelength_range=None)

fig, ax = plt.subplots(2, 1, figsize=(6.35, 3.5), sharex="col")
ax[0].plot(wave, spec,  lw=0.2)
ax[0].plot(wave, planck(wave*1e-10, T=T)*10*np.pi)

print(wave.shape)
ax[0].set_xlim(0, 25000)
ax[0].set_ylim(0, ax[0].get_ylim()[1])

fig.set_tight_layout(True)
plt.savefig("dbug.png", dpi=600)
