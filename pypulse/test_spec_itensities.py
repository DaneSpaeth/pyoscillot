from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from dataloader import phoenix_spectrum, carmenes_template

with fits.open("lte06000-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits") as hdul:
    print(hdul.info())
    header = hdul[0].header
    # si: specific intensity
    si_spectrum = np.array(hdul[0].data)
    mu = hdul[1].data

si_wave = np.arange(500, 25500 + 500, 1.0)
print(mu)
idx = np.where(mu == 0.25675)[0][0]
print(idx)

si_spec = si_spectrum[idx, :]
wave, spec, header = phoenix_spectrum(
   6000, 4.5, 0.0, wavelength_range=(500, 25500 + 500))

#spec = spec / np.mean(spec)
si_spec = si_spec / np.mean(si_spec)
# plt.plot(wave, spec, color="blue", label="NORMAL")
plt.plot(si_wave, si_spec, color="red", label="SPECIFIC")
# plt.show()

print(wave)

# plt.show()

# (spec, cont, sig, wave) = carmenes_template()
# print(wave)
