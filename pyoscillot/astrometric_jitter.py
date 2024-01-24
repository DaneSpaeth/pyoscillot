import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from plapy.constants import SUN_RAD, PARSEC


def calc_photocenter(intensity_2d):
    """ Calculate the photocenter_position for a 2d intensity map."""
    # Scipy has the image coordinates flipped
    center_of_mass = ndimage.center_of_mass(intensity_2d)
    return center_of_mass[1], center_of_mass[0]

def calc_astrometric_deviation(diff_photocenter, N_star, R_star=1, distance_pc=10):
    """ Calculate the astrometric deviation given the difference of the photocenter from the
        geometric photocenter in px, the number of pixels of the star in the simulation and
        the distance in pc
    """
    abs_diff_px = np.sqrt(diff_photocenter[0]**2+diff_photocenter[1]**2)
    abs_diff_pct = abs_diff_px / N_star
    abs_diff = abs_diff_pct * R_star

    astrometric_deviation = abs_diff * SUN_RAD / (distance_pc * PARSEC)
    astrometric_deviation_mas = np.degrees(astrometric_deviation) * 3600e6

    return astrometric_deviation_muas

def convert_to_muas(value_px, N_star, R_star, distance_pc):
    """ Convert a value in pixel to microacsecs"""
    value_pct = value_px / N_star
    abs_value = value_pct * R_star

    value_rad = abs_value * SUN_RAD / (distance_pc * PARSEC)
    value_mas = np.degrees(value_rad) * 3600e6

    return value_mas

def calc_for_timeseries():
    from pathlib import Path
    directory = Path("/home/dspaeth/data/simulations/fake_spectra/GRANULATION_HIGHRES/arrays/intensity_stefan_boltzmann")
    files = sorted(list(directory.glob("*.npy")))
    fig, ax = plt.subplots(1)
    for file in files:
        intensity_2d = np.load(file)
        geom_center = (np.array(intensity_2d.shape) - 1) / 2
        photocenter_xy = np.array(calc_photocenter(intensity_2d))
        diff_photocenter = photocenter_xy - geom_center
        ax.scatter(convert_to_muas(diff_photocenter[0], 500, 1, 10),
                   convert_to_muas(diff_photocenter[1], 500, 1, 10), marker="x", color="blue")
        ax.scatter(0,0, marker="o", color="red")

    ax.set_title("Astrometric Jitter, Assumes R=1R_sun, d=10pc")
    ax.set_xlabel("Deviation X [µas]")
    ax.set_ylabel("Deviation Y [µas]")

    plt.savefig("/home/dspaeth/data/simulations/tmp_plots/astrometric_jitter.png", dpi=300)

    plt.show()


    radius = 1
    distance = 10






if __name__ == "__main__":
    calc_for_timeseries()