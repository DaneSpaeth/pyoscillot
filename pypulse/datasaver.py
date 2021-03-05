from astropy.io import fits
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from shutil import copy2
DATAROOT = Path("/home/dane/Documents/PhD/pypulse/data")


def save_spectrum(spectrum, new_header, name):
    """ Save a Carmenes spectrum from spectrum."""
    template = DATAROOT / "template.fits"
    outfile = DATAROOT / "fake_spectra" / f"{name}.fits"

    print(f"Copy template to {outfile}")
    copy2(template, outfile)

    with fits.open(outfile, mode="update") as hdul:
        # First fix the index error
        for i in range(0, len(hdul)):
            hdul[i].verify("fix")

        # Now update the primary header
        for key, value in new_header.items():
            if key in hdul[0].header.keys():
                hdul[0].header[key] = value

        # TODO add saving of spec
        print(hdul[1].data.shape)
        print(spectrum.shape)
        hdul[1].data = spectrum
        hdul.flush()


if __name__ == "__main__":

    # save_spectrum(None, None, {"OBJECT": "TEST"}, "test")
    template = DATAROOT / "template.fits"
    with fits.open(template) as hdul:
        print(hdul.info())
        fake_spec = np.zeros((61, 4096))
        spec = hdul[1].data
        cont = hdul[2].data
        sig = hdul[3].data
        wave = hdul[4].data

    plt.plot(wave[20], spec[20])
    plt.show()
    #     hdul.flush()

    # idx = 30
    # plt.plot(wave[idx], spec[idx])
    # plt.show()
