import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import plapy.rv.dataloader as load
from parse_ini import parse_global_ini


def check_time_series(name):
    """ Plot a check plot.

            TODO: Change it to use the usual plot functions.
    """
    # Read in the flux
    global_dict = parse_global_ini()
    rvlibpath = global_dict["rvlibpath"]
    fluxfile = rvlibpath / name / "flux.txt"
    bjd = []
    flux = []
    with open(fluxfile, "r") as f:
        for line in f:
            columns = line.strip().split()
            bjd.append(float(columns[0]))
            flux.append(float(columns[1]))
    bjd = np.array(bjd)
    flux = np.array(flux)
    flux = flux / np.median(flux)

    # Read in RV, CRX and DLW
    rv_dict = load.rv(name)
    crx_dict = load.crx(name)["SIMULATION"]
    dlw_dict = load.dlw(name)["SIMULATION"]

    time = rv_dict["SIMULATION"][0]
    rv = rv_dict["SIMULATION"][1]
    rve = rv_dict["SIMULATION"][2]
    print(len(flux))
    print(len(rv))

    # PLOT
    BJD_OFFSET = 2400000.5
    fig, ax = plt.subplots(3, 2, figsize=(20, 10))

    ax[0, 0].errorbar(time - BJD_OFFSET, rv, yerr=rve,
                      linestyle="None", marker="o")
    ax[0, 0].set_xlabel("Time [MJD]")
    ax[0, 0].set_ylabel("RV [m/s]")
    ax[0, 1].plot(bjd - BJD_OFFSET, flux, linestyle="None", marker="o")
    ax[0, 1].set_xlabel("Time [MJD]")
    ax[0, 1].set_ylabel("Flux [%]")

    ax[1, 0].errorbar([b - BJD_OFFSET for b in crx_dict["bjd"]], crx_dict["crx"],
                      yerr=crx_dict["crxe"], linestyle="None", marker="o")
    ax[1, 0].set_xlabel("Time [MJD]")
    ax[1, 0].set_ylabel("CRX [m/s/Np]")
    ax[2, 0].errorbar(rv, crx_dict["crx"],
                      yerr=crx_dict["crxe"], linestyle="None", marker="o")
    ax[2, 0].set_xlabel("RV [m/s]")
    ax[2, 0].set_ylabel("CRX [m/s/Np]")
    ax[1, 1].errorbar([b - BJD_OFFSET for b in dlw_dict["bjd"]], dlw_dict["dlw"],
                      yerr=dlw_dict["dlwe"], linestyle="None", marker="o")
    ax[1, 1].set_xlabel("Time [MJD]")
    ax[1, 1].set_ylabel("dlW [m^2/s^2]")
    # ax[1].set_ylim(-100, 100)
    ax[2, 1].errorbar(rv, dlw_dict["dlw"],
                      yerr=dlw_dict["dlwe"], linestyle="None", marker="o")
    ax[2, 1].set_xlabel("RV [m/s]")
    ax[2, 1].set_ylabel("dLW [m^²/s^2]")
    # fig.suptitle(name)
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
    plot_temperature("test_arrays")
