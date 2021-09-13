from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

with fits.open("lte06000-4.50-0.0.PHOENIX-ACES-AGSS-COND-SPECINT-2011.fits") as hdul:
    print(hdul.info())
    header = hdul[0].header
    # si: specific intensity
    si_spectrum = np.array(hdul[0].data)
    mu = hdul[1].data

plt.plot(si_spectrum[10, :])
plt.show()


print(mu)
