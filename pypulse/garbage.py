import matplotlib.pyplot as plt
from dataloader import phoenix_spectrum


rest_wavelength, rest_spectrum, _ = phoenix_spectrum(
    Teff=4800, wavelength_range=(3000, 10000))
plt.plot(rest_wavelength, rest_spectrum)
plt.xlim((3000, 10000))
plt.ylim(0, rest_spectrum.max() + 10000)
plt.xlabel("Wavelength [Angstrom]")
plt.ylabel("Flux")
plt.tight_layout()
plt.show()
