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


if __name__ == "__main__":
    file = Path(
        "/home/dane/Documents/PhD/pypulse/data/fake_serval/HIP73620_vis/HIP73620_vis.rvc.dat")
    time, rv, rve = read_in_rvs(file)

    plt.scatter(time, rv)
    plt.show()
