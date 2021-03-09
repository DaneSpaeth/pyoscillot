from astropy.io import fits
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from shutil import copy2
DATAROOT = Path("/home/dane/Documents/PhD/pypulse/data")


def save_spectrum(spectrum, new_header, name):
    """ Save a Carmenes spectrum from spectrum."""
    template = DATAROOT / "template.fits"
    if not name.endswith("fits"):
        name += ".fits"

    outfile = DATAROOT / "fake_spectra" / name

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

    from scipy.signal import medfilt
    # save_spectrum(None, None, {"OBJECT": "TEST"}, "test")
    template = DATAROOT / "template.fits"
    with fits.open(template) as hdul:
        header = hdul[0].header
        spec = hdul[1].data
        cont = hdul[2].data
        sig = hdul[3].data
        wave = hdul[4].data

    for key in header:
        print(key)

    exit()

    fake = DATAROOT / "fake_spectra" / "car-20190717T00h00m00s-sci-fake-vis_A.fits"
    with fits.open(fake) as hdul:
        fspec = hdul[1].data
        fcont = hdul[2].data
        fsig = hdul[3].data
        fwave = hdul[4].data

    order = 40
    plt.plot(wave[order], fspec[order])
    plt.plot(wave[order], spec[order])

    N = 501
    rolling_avg = np.convolve(spec[order], np.ones(N) / N, mode='same')
    rolling_med = medfilt(spec[order], kernel_size=N)
    # plt.plot(wave[order], fspec[order] * rolling_med / np.mean(rolling_med))
    plt.show()
    #     hdul.flush()

    # idx = 30
    # plt.plot(wave[idx], spec[idx])
    # plt.show()
