from pathlib import Path
import numpy as np
from datasaver import read_in
import matplotlib.pyplot as plt
from utils import bisector_new

root = Path("/home/dane/Documents/PhD/pyCARM/data/by_hip/HIP73620")

if __name__ == "__main__":
    files = root.glob("*vis_A.fits")
    for f in files:
        header, spec, cont, sig, wave = read_in(f)

        order = 40

        wave = wave[order]
        spec = spec[order]

        w_mask = np.logical_and(
            wave > 7799.6, wave < 7800.2)
        wave = wave[w_mask]
        spec = spec[w_mask]

        spec = spec / np.max(spec)

        bis_wave, bis, center = bisector_new(wave, spec, skip=0)
        # plt.plot(wave, spec, "bo")
        plt.plot(bis_wave, bis)
        plt.show()
        exit()
