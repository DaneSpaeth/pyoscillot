from astropy.io import fits
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from shutil import copy2
from scipy.ndimage import gaussian_filter1d


# DATAROOT = Path("/home/dane/Documents/PhD/pypulse/data")
DATAROOT = Path(__file__).parent.parent / "data"

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


def read_in(file):
    with fits.open(file) as hdul:
        header = hdul[0].header
        spec = hdul[1].data
        cont = hdul[2].data
        sig = hdul[3].data
        wave = hdul[4].data

    return header, spec, cont, sig, wave


if __name__ == "__main__":

    from utils import adjust_snr
    # save_spectrum(None, None, {"OBJECT": "TEST"}, "test")
    sim = DATAROOT / "car-20171205T00h00m00s-sci-fake-vis_A.fits"
    template = DATAROOT / "template.fits"

    header, spec, cont, sig, wave = read_in(sim)
    t_header, t_spec, t_cont, t_sig, t_wave = read_in(template)

    idx = 33

    snr = spec / sig
    t_snr = t_spec / t_sig

    # plt.plot(wave[33], spec[33])
    fig, ax = plt.subplots(1, 3)
    ax[0].plot(wave[idx], spec[idx], label=f"Simulated order {idx}")
    ax[0].plot(t_wave[idx], t_spec[idx], label=f"Template order {idx}")
    ax[1].plot(t_wave[idx], t_sig[idx], label=f"Template order {idx}")
    ax[1].plot(wave[idx], sig[idx], label=f"Simulated order {idx}")
    ax[2].plot(wave[idx], snr[idx], label=f"Simulated order {idx}")
    ax[2].plot(t_wave[idx], t_snr[idx], label=f"Template order {idx}")
    plt.legend()
    plt.show()
