import matplotlib.pyplot as plt
from pathlib import Path
import astropy.io.fits as fits
import numpy as np
import plapy.rv.dataloader as load
import shutil

RAW_DIRECTORY = Path("/home/dane/Documents/PhD/pyCARM/data/by_hip")
OUT_DIRECTORY = Path("/home/dane/Documents/PhD/pyoscillot/data/")
stars = load.star_list()


def get_snr_per_order(file):
    """ Return an array of the median snr per order for one file."""
    nr_orders = 61
    snr_per_order = np.zeros(nr_orders)
    with fits.open(file) as hdul:

        header = hdul[0].header
        for order in range(nr_orders):
            snr_per_order[order] = float(header[f"CARACAL FOX SNR {order}"])
    return snr_per_order


for star in stars:
    star_dir = RAW_DIRECTORY / star
    files = star_dir.glob("*vis_A.fits")
    star_snr_per_order = []
    max_snr_file = None
    max_snr = 0
    for file in files:
        snr_per_order = get_snr_per_order(file)
        star_snr_per_order.append(snr_per_order / np.max(snr_per_order))
        if np.max(snr_per_order) > max_snr:
            max_snr = np.max(snr_per_order)
            max_snr_file = file

    star_snr_per_order = np.array(star_snr_per_order)

    mean_snr_per_order = np.mean(star_snr_per_order, axis=0)

    # Create the SNR profile for the star
    np.save(OUT_DIRECTORY / "CARMENES_SNR_profiles" / star, mean_snr_per_order)

    # Copy the highest SNR template to the templates folder
    # Not sure if that is really useful but let's do it anyway
    shutil.copy2(max_snr_file,
                 OUT_DIRECTORY / "CARMENES_templates" / f"CARMENES_template_{star}.fits")
