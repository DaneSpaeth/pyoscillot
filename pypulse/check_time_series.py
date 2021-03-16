import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def read_in_rvs(file):
    """ Return np.arrays of time, rv and rve.

        :param file: File to read data from.
    """
    time = []
    rv = []
    rve = []

    with open(file, "r") as f:
        for line in f:
            line = line.strip()
            columns = line.split()
            time.append(float(columns[0]))
            rv.append(float(columns[1]))
            rve.append(float(columns[2]))

    time = np.array(time)
    rv = np.array(rv)
    rve = np.array(rve)

    return time, rv, rve


def _read_in_crx(file):
    """ Read in info from crx.dat file."""
    crx_dict = {"bjd": [], "crx": [], "crxe": [],
                "crx_off": [], "crx_offe": [], "l_v": [], "logwave": {}}
    with open(file, "r") as f:
        for idx, line in enumerate(f):
            columns = line.strip().split()
            zeroth_order_offset = 6
            for key_idx, key in enumerate(crx_dict.keys()):
                if key != "logwave":
                    crx_dict[key].append(float(columns[key_idx]))

                for col_nr in range(zeroth_order_offset, len(columns)):
                    i = col_nr - zeroth_order_offset
                    if idx == 0:
                        crx_dict["logwave"][i] = []
                    crx_dict["logwave"][i].append(float(columns[col_nr]))

    return crx_dict


if __name__ == "__main__":
    root = Path(
        "/home/dane/Documents/PhD/pypulse/data/fake_serval/HIP73620_vis/")
    file = root / "HIP73620_vis.rvc.dat"
    time, rv, rve = read_in_rvs(file)
    crx_file = root / "HIP73620_vis.crx.dat"
    crx_dict = _read_in_crx(crx_file)

    print(crx_dict.keys())

    fig, ax = plt.subplots(3)

    ax[0].scatter(time - 2400000.5, rv)
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("RV [m/s]")
    ax[1].errorbar([b - 2400000.5 for b in crx_dict["bjd"]], crx_dict["crx"],
                   yerr=crx_dict["crxe"], linestyle="None", marker="o")
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("CRX [m/s/Np]")
    ax[2].errorbar(rv, crx_dict["crx"],
                   yerr=crx_dict["crxe"], linestyle="None", marker="o")
    ax[2].set_xlabel("RV [m/s]")
    ax[2].set_ylabel("CRX [m/s/Np]")
    # ax[0].set_title("Pulsation, vsini=0, v_p=400m/s, no spot, no temp variation")
    plt.show()
