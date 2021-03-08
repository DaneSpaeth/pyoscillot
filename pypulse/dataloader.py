from astropy.io import fits
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
DATAROOT = Path("/home/dane/Documents/PhD/pypulse/data")


def phoenix_spectrum(Teff=4500, logg=2.5, feh=-0.5, wavelength_range=(3000, 7000)):
    """Return phenix spectrum and header."""
    file = DATAROOT / "phoenix_spectra" /\
        f"lte0{Teff}-{logg:.2f}{feh}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
    with fits.open(file) as hdul:
        header = hdul[0].header
        spectrum = hdul[0].data

    wavelength_file = DATAROOT / "phoenix_spectra" / \
        "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"

    with fits.open(wavelength_file) as hdul:
        wavelength = hdul[0].data

    if wavelength_range:
        wavelength_mask = np.logical_and(wavelength >= wavelength_range[0],
                                         wavelength <= wavelength_range[1],)
        spectrum = spectrum[wavelength_mask]
        wavelength = wavelength[wavelength_mask]

    return spectrum, wavelength, header


def carmenes_template():
    """ Return spec, sig, cont and wave of Carmenes template."""
    template = DATAROOT / "template.fits"
    with fits.open(template) as hdul:
        spec = hdul[1].data
        cont = hdul[2].data
        sig = hdul[3].data
        wave = hdul[4].data

    return (spec, cont, sig, wave)


if __name__ == "__main__":
    wavelength_range = (3000, 30000)
    spectrum, wavelength, _ = load_spectrum(wavelength_range=wavelength_range)
    spectrum_low, wavelength, _ = load_spectrum(
        Teff=3000, wavelength_range=wavelength_range)
    plt.plot(wavelength, spectrum)
    plt.plot(wavelength, spectrum_low)
    plt.show()
