import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import plapy.rv.dataloader as load
from parse_ini import parse_global_ini
import sys
try:
    sys.path.append("/home/dane/Documents/PhD/pyCARM/pyCARM")
    from plotter import plot_rv, plot_activity, plot_activity_rv
except ModuleNotFoundError:
    pass


def check_time_series(name, instrument=None):
    """ Plot a check plot.

            TODO: Change it to use the usual plot functions.
    """
    # Read in the flux
    global_dict = parse_global_ini()
    rvlibpath = global_dict["rvlibpath"]
    fluxfile = rvlibpath / name / "flux.txt"
    bjd = []
    flux = []
    try:
        with open(fluxfile, "r") as f:
            for line in f:
                columns = line.strip().split()
                bjd.append(float(columns[0]))
                flux.append(float(columns[1]))
    except FileNotFoundError:
        pass
    bjd = np.array(bjd)
    flux = np.array(flux)
    flux = flux / np.median(flux)

    # Read in RV, CRX and DLW
    fig, ax = plt.subplots(3, 2, figsize=(20, 10))
    rv_dict = load.rv(name)
    crx_dict = load.crx(name)
    dlw_dict = load.dlw(name)

    plot_rv(rv_dict, ax=ax[0, 0], instrument=instrument)
    plot_activity(crx_dict, ax=ax[1, 0], instrument=instrument)
    plot_activity(dlw_dict, ax=ax[2, 0], instrument=instrument)
    plot_activity_rv(rv_dict, crx_dict, ax=ax[1, 1], instrument=instrument)
    plot_activity_rv(rv_dict, dlw_dict, ax=ax[2, 1], instrument=instrument)
    fig.tight_layout()
    plt.show()


def plot_temperature(name):
    """ Plot one array of the temperature maps.

        In the future plot it as a gif.
    """
    # Read in the flux
    global_dict = parse_global_ini()
    datapath = global_dict["datapath"]

    temperature_folder = datapath / "fake_spectra" / name / "arrays" / "temperature"
    array_paths = list(temperature_folder.glob("*.npy"))

    array = np.load(array_paths[0])
    plt.imshow(array, origin="lower", cmap="seismic")
    plt.show()


if __name__ == "__main__":
    check_time_series("talk_ngc2423_90", instrument="CARMENES_VIS")
    # check_time_series("NGC2423", instrument="HARPS")
